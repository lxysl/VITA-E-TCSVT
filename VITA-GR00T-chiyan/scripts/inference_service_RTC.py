# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import numpy as np

from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy_vita import Gr00tPolicy



import time
from collections import deque
from typing import Dict, Any
import copy
import threading

PREDICT_HORIZON = 16          # == H
EXECUTE_HORIZON = 8           # == s_init
MIN_EXEC_HORIZON = 6         # == s_min
DELAY_BUFFER     = 6         # == b
CTRL_PERIOD_SEC  = 0.05      # == Δt  (50ms控制回环，控制频率20Hz) 25Hz会跟不上
D_INIT = 5
class RealTimeChunkController:
    """
    前台:  controller.step(obs)  每个控制周期调用一次，立即返回单步动作
    后台:  独立线程，异步调用 policy_client.get_realtime_action(...)
    """
    def __init__(self,
                 policy_client: RobotInferenceClient,
                 prediction_horizon: int = PREDICT_HORIZON,
                 min_exec_horizon: int = MIN_EXEC_HORIZON,
                 delay_buf_size: int = DELAY_BUFFER,
                 ctrl_period: float = CTRL_PERIOD_SEC,
                 d_init: int = D_INIT,
                 s_init: int = EXECUTE_HORIZON):

        self.client = policy_client
        self.H     = prediction_horizon
        self.s_min = min_exec_horizon
        self.ctrl_period = ctrl_period
        self.d_init = d_init
        self.s_init = s_init
        self.delay_buf_size = delay_buf_size
        
        # ------------- 共享状态 (互斥保护) -------------
        self.t: int                       = 0          # 块内游标
        self.A_cur: np.ndarray | None     = None       # 当前动作块, shape (H, *action_shape)
        self.prev_action_chunk: Any       = None       # 传回服务器的 prev_action_chunk
        self.o_cur: Dict[str, Any] | None = None       # 最近观测

        self.Q = deque([self.d_init], maxlen=self.delay_buf_size)     # 延迟缓冲，先放一个5防止空

        self._infer_in_flight: bool = False
        # 同步原语
        self.M = threading.Lock()
        self.C = threading.Condition(self.M)

        # 后台推理线程
        self._infer_th = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_th.start()
        self.last_time=time.time()
        
    def reset(self):
        with self.C:
            # 等待后台线程不在 RPC 状态
            while self._infer_in_flight:
                self.C.wait()
            self.t = 0          # 块内游标
            self.A_cur = None       # 当前动作块, shape (H, *action_shape)
            self.prev_action_chunk = None       # 传回服务器的 prev_action_chunk
            self.o_cur = None       # 最近观测
            self.Q.clear()
            self.Q.append(self.d_init)
            
            # 唤醒后台线程，防止它还在 wait() 卡住
            self.C.notify_all()
        
    # --------------------------------------------------
    # 前台：每 Δt 调用一次
    # --------------------------------------------------
    def step(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        调用者(控制器/主线程)每个控制周期执行一次。
        obs 必须是发送给服务器的同构字典 (可缺 prev_action_chunk；本函数会补)
        返回 单步动作 (Dict[str, np.ndarray]) —— 直接可发给机器人执行
        """
        # 注：为了防止外部代码在后台推理时修改 obs，这里只放浅拷贝进共享区
        obs_for_shared = dict(obs)
        with self.C:
            # ========== 冷启动：若还没有动作块就同步要一块，以保证第一步能控制 ==========
            if self.A_cur is None:
                obs_for_shared["prev_action_chunk"] = self.prev_action_chunk
                obs_for_shared["inference_delay"] = self.d_init
                obs_for_shared["execute_horizon"] = self.s_init
                bootstrap = self.client.get_realtime_action(obs_for_shared)
                self.A_cur = bootstrap["action.actions"]
                self.prev_action_chunk = bootstrap["prev_action_chunk"]

            # ========== 常规 GETACTION ==========
            self.t += 1
            self.o_cur = obs_for_shared              # 更新最新观测
            self.C.notify()                          # 唤醒后台线程（若在 wait）

            if self.t-1 >= len(self.A_cur):
                single_action = self.A_cur[-1]
                print("已耗尽")
            else:
                single_action = self.A_cur[self.t - 1]
        return single_action

    # --------------------------------------------------
    # 后台线程：INFERENCELOOP
    # --------------------------------------------------
    def _inference_loop(self):
        while True:
            with self.C:
                # 1) 等到前台已执行 >= s_min 个动作
                while self.t < self.s_min:
                    self.C.wait()

                # ----------- 临界区（复制必要数据） -----------
                s   = self.t                                        # 已执行步数
                o   = copy.deepcopy(self.o_cur)                    # 深拷贝，防止并发写
                d   = max(self.Q)                                  # 延迟估计
                o["prev_action_chunk"] = self.prev_action_chunk
                o["inference_delay"] = d
                o["execute_horizon"] = s
                
                # 2) 调用服务器进行推理（GUIDEDINFERENCE封装在get_realtime_action内）
                self._infer_in_flight = True
                self.C.release()  # 释放锁
                realtime_action = self.client.get_realtime_action(o)
                self.C.acquire()  # 获取锁

                # 3) 把新动作块写回共享状态
                self.A_cur = realtime_action["action.actions"]
                self.prev_action_chunk = realtime_action["prev_action_chunk"]

                self.t = self.t - s
                self.Q.append(self.t)             # 记录延迟
                self._infer_in_flight = False
                self.C.notify_all()                # 万一前台在 wait
                # print(f"[inference]  latency={time.time()-self.last_time:.4f}s  s={s}  d={d}  self.t={self.t}")
                # self.last_time=time.time()

def vita_gr00t_client(args):
    policy_client = RobotInferenceClient(host=args.host, port=args.port)

    print("Available modality config available:")
    modality_configs = policy_client.get_modality_config()
    print(modality_configs.keys())

    # 构造实时分块控制器
    controller = RealTimeChunkController(policy_client)
    next_t = time.perf_counter()
    while True:
        # prev_action_chunk = None
        obs = {
            "video.image": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
            "video.wrist_image": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
            "state.state": np.expand_dims(
                np.concatenate(
                    (
                        np.random.rand(3),
                        np.random.rand(3),
                        np.random.rand(2),
                    )
                ),
                axis=0
            ),
            "annotation.human.task_description": ["pick up the apple."],
        }

        action = controller.step(obs)
        
        # realtime_action = policy_client.get_realtime_action(obs, inference_delay=0, execute_horizon=8)
        # action_chunk=realtime_action["action.actions"]
        # prev_action_chunk=realtime_action["prev_action_chunk"]
        
        
        next_t += CTRL_PERIOD_SEC
        # 距离下一次开始还剩多少时间
        sleep_time = next_t - time.perf_counter()
        print(sleep_time)
        if sleep_time > 0:
            time.sleep(sleep_time)     # 足够快，按计划睡
        else:
            next_t = time.perf_counter()
        
        controller.reset()  # Important!!! need to reset obs at the beginning of each task


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        # default="/root/Isaac-GR00T/checkpoints/vita_libero_90_fixed_bs64_gpu8_frozen_vlm_fix_lang/checkpoint-28000",
        default="/root/Isaac-GR00T/checkpoints/finetuned_vita_libero_pretrained_finetuned_mixed_bs64_gpu8/checkpoint-16000",
        # default="/root/Isaac-GR00T/checkpoints/finetuned_vita_libero_90_10_fixed_bs64_gpu8_frozen_vlm/checkpoint-6000",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="new_embodiment",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        help="The name of the data config to use.",
        choices=list(DATA_CONFIG_MAP.keys()),
        default="libero_vita",
    )

    parser.add_argument("--port", type=int, help="Port number for the server.", default=8005)
    parser.add_argument(
        "--host", type=str, help="Host address for the server.", default="0.0.0.0"
    )
    # server mode
    parser.add_argument("--server", action="store_true", help="Run the server.")
    # client mode
    parser.add_argument("--client", action="store_true", help="Run the client")
    parser.add_argument("--denoising_steps", type=int, help="Number of denoising steps.", default=4)
    args = parser.parse_args()

    if args.server:
        data_config = DATA_CONFIG_MAP[args.data_config]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        print("loading model")
        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
        )
        print("model loaded!")
        # Start the server
        server = RobotInferenceServer(policy, port=args.port)
        server.run()

    elif args.client:
        vita_gr00t_client(args)
    else:
        raise ValueError("Please specify either --server or --client")