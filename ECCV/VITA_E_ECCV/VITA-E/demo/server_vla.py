from __future__ import print_function

import argparse
import asyncio
import base64
import builtins
import datetime
import io
import json
import multiprocessing
import os
import re
import threading
import time
import warnings
from queue import Empty
from typing import AsyncGenerator
from threading import Timer
from collections import deque
import cv2
import pickle
import numpy as np
from PIL import Image
import traceback

from flask import Flask, current_app, render_template, request
from flask_socketio import SocketIO, disconnect, emit

from web_demo.vita_html.web.parms import GlobalParams
from web_demo.vita_html.web.pem import generate_self_signed_cert
from web_demo.gr00t.experiment.data_config import DATA_CONFIG_MAP
from web_demo.gr00t.model.policy_vita_action_head import Gr00tActionHeadPolicy

# 导入工具模块
from web_demo.server_utils import (
    Colors, ProcessConfig, 
    IMAGE_TOKEN_INDEX, AUDIO_TOKEN_INDEX, IMAGE_TOKEN, AUDIO_TOKEN, VIDEO_TOKEN,
    AUDIO_RESPONSE_TOKEN, TEXT_RESPONSE_TOKEN, ACT_TOKEN, HALT_TOKEN, 
    INSTRUCTION_TOKEN, ACTION_END_TOKEN, TEXT_END_TOKEN, SPECIAL_TOKEN_LIST,
    decoder_topk, codec_padding_size, target_sample_rate,
    load_model_embemding, save_video, tokenizer_image_audio_token, 
    clear_queue, merge_current_and_history
)

def get_args():
    parser = argparse.ArgumentParser(description='VITA')
    parser.add_argument('--model_path', help='model_path to load', default='./vla_demo_VITA_ckpt_ended')
    parser.add_argument('--gr00t_model_path', help='GR00T model path for action generation', default='/root/Isaac-GR00T/checkpoints/vita_real_data_bs64_gpu8_frozen_vlm_robot/checkpoint-15000')
    parser.add_argument('--data_config_name', help='Data config name', default='real_data_robot_vita_action_head')
    parser.add_argument('--ip', help='ip of server', default='127.0.0.1')
    parser.add_argument('--port', help='port of server', default=8081)
    parser.add_argument('--max_users', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    print(f"{Colors.CYAN}VITA server args: {args}{Colors.RESET}")
    return args

def custom_print(*args, **kwargs):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    original_print(f'[{current_time}]', *args, **kwargs)

args = get_args()
last_tts_model_id = 0

# change print function to add time stamp
original_print = builtins.print
builtins.print = custom_print

# init flask app
app = Flask(__name__, template_folder='./vita_vla_html/web/resources', static_folder='./vita_vla_html/web/static')
socketio = SocketIO(app)
# init connected users
connected_users = {}

def disconnect_user(sid):
    if sid in connected_users:
        print(f"{Colors.RED}Disconnecting user {sid} due to time out{Colors.RESET}")
        socketio.emit('out_time', to=sid) 
        connected_users[sid][0].cancel()
        connected_users[sid][1].interrupt()
        connected_users[sid][1].stop_pcm = True
        connected_users[sid][1].release()
        time.sleep(3)
        del connected_users[sid]

def load_model(
        llm_id,
        engine_args,
        cuda_devices,
        inputs_queue,
        outputs_queue,
        tts_outputs_queue,
        action_feature_queue,
        action_feature_stack,
        stop_event,
        other_stop_event,
        worker_ready,
        wait_workers_ready,
        start_event,
        other_start_event,
        start_event_lock,
        is_generating_action_feature,
        other_is_generating_action_feature,
        current_dialog,
        global_history,
        shutdown_event,
        global_history_limit=0,
        gr00t_model_path=None,
        data_config_name=None,
        observation_queue=None,
        observation_queue_lock=None,
    ):
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    
    # 导入依赖CUDA的包
    import torch
    import torchaudio
    from vllm import LLM, SamplingParams
    from transformers import AutoFeatureExtractor, AutoTokenizer
    from decord import VideoReader, cpu
    from vita.model.language_model.vita_qwen2 import VITAQwen2Config, VITAQwen2ForCausalLM
    
    #等待tts初始化
    print(f"{Colors.MAGENTA}wait_workers_ready: {wait_workers_ready}{Colors.RESET}")
    wait_workers_ready[1].wait()
    print(f"{Colors.MAGENTA}wait_workers_ready status updated: {wait_workers_ready}{Colors.RESET}")
    llm = LLM(
            model=engine_args,
            dtype="float16",
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.4,  # 调整以解决 OOM 或减少显存使用
            disable_custom_all_reduce=True,
            limit_mm_per_prompt={'image':256,'audio':50}
        )

    tokenizer = AutoTokenizer.from_pretrained(engine_args, trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(engine_args, subfolder="feature_extractor", trust_remote_code=True)

    sampling_params = SamplingParams(temperature=0.01, max_tokens=512, best_of=1, skip_special_tokens=False)
    
    # 初始化GR00T action policy
    print(f"{Colors.CYAN}Initializing GR00T action policy from {gr00t_model_path}{Colors.RESET}")
    data_config = DATA_CONFIG_MAP[data_config_name]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    gr00t_action_policy = Gr00tActionHeadPolicy(
        model_path=gr00t_model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag="new_embodiment",
        denoising_steps=4,
    )
    print(f"{Colors.CYAN}GR00T action policy initialized successfully{Colors.RESET}")

    def _process_inputs(inputs):

        def _process_image(image_path):
            if isinstance(image_path, str):
                assert os.path.exists(image_path), f"Image file {image_path} does not exist."
                return Image.open(image_path).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
            else:
                assert isinstance(image_path, np.ndarray), "Image must be either a file path or a numpy array."
                return Image.fromarray(image_path).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)


        def _process_audio(audio_path):
            assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist."
            audio, sr = torchaudio.load(audio_path)
            audio_features = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")["input_features"]
            audio_features = audio_features.squeeze(0)
            return audio_features
        
        def _process_video(video_path, max_frames=4, min_frames=4, s=None, e=None):
            # speed up video decode via decord.

            if s is None or e is None:
                start_time, end_time = None, None
            else:
                start_time = int(s)
                end_time = int(e)
                start_time = max(start_time, 0)
                end_time = max(end_time, 0)
                if start_time > end_time:
                    start_time, end_time = end_time, start_time
                elif start_time == end_time:
                    end_time = start_time + 1

            if os.path.exists(video_path):
                vreader = VideoReader(video_path, ctx=cpu(0))
            else:
                raise FileNotFoundError(f"Video file {video_path} does not exist.")

            fps = vreader.get_avg_fps()
            f_start = 0 if start_time is None else int(start_time * fps)
            f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
            num_frames = f_end - f_start + 1
            
            if num_frames > 0:
                # T x 3 x H x W
                all_pos = list(range(f_start, f_end + 1))
                if len(all_pos) > max_frames:
                    sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
                elif len(all_pos) < min_frames:
                    sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)]
                else:
                    sample_pos = all_pos

                # patch_images = [Image.fromarray(f).transpose(Image.FLIP_LEFT_RIGHT) for f in vreader.get_batch(sample_pos).asnumpy()]
                patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
                return patch_images

            else:
                print(f"{Colors.RED}video path: {video_path} error.{Colors.RESET}")

        if "multi_modal_data" in inputs:

            if "image" in inputs["multi_modal_data"]:
                image_inputs = inputs["multi_modal_data"]["image"]
                if not isinstance(image_inputs, list):
                    image_inputs = [image_inputs]
                inputs["multi_modal_data"]["image"] = [_process_image(f) for f in image_inputs]

                if "prompt" in inputs:
                    assert inputs["prompt"].count(IMAGE_TOKEN) == len(image_inputs), \
                        f"Number of image token {IMAGE_TOKEN} in prompt must match the number of image inputs."
                elif "prompt_token_ids" in inputs:
                    assert inputs["prompt_token_ids"].count(IMAGE_TOKEN_INDEX) == len(image_inputs), \
                        f"Number of image token ids {IMAGE_TOKEN_INDEX} in prompt_token_ids must match the number of image inputs."
                else:
                    raise ValueError("Either 'prompt' or 'prompt_token_ids' must be provided.")

            if "audio" in inputs["multi_modal_data"]:
                audio_inputs = inputs["multi_modal_data"]["audio"]
                if not isinstance(audio_inputs, list):
                    audio_inputs = [audio_inputs]
                inputs["multi_modal_data"]["audio"] = [_process_audio(f) for f in audio_inputs]

                if "prompt" in inputs:
                    assert inputs["prompt"].count(AUDIO_TOKEN) == len(inputs["multi_modal_data"]["audio"]), \
                        f"Number of audio token {AUDIO_TOKEN} in prompt must match the number of audio inputs."
                elif "prompt_token_ids" in inputs:
                    assert inputs["prompt_token_ids"].count(AUDIO_TOKEN_INDEX) == len(inputs["multi_modal_data"]["audio"]), \
                        f"Number of audio token ids {AUDIO_TOKEN_INDEX} in prompt_token_ids must match the number of audio inputs."
                else:
                    raise ValueError("Either 'prompt' or 'prompt_token_ids' must be provided.")

            if "video" in inputs["multi_modal_data"]:
                video_inputs = inputs["multi_modal_data"]["video"]
                if not isinstance(video_inputs, list):
                    video_inputs = [video_inputs]

                assert "prompt" in inputs, "Prompt must be provided when video inputs are provided."
                assert "image" not in inputs["multi_modal_data"], "Image inputs are not supported when video inputs are provided."

                assert inputs["prompt"].count(VIDEO_TOKEN) == 1, "Currently only one video token is supported in prompt."

                assert inputs["prompt"].count(VIDEO_TOKEN) == len(inputs["multi_modal_data"]["video"]), \
                    f"Number of video token {VIDEO_TOKEN} in prompt must match the number of video inputs."
                
                video_frames_inputs = []
                for video_input in video_inputs:
                    video_frames_inputs.extend(_process_video(video_input, max_frames=4, min_frames=4))
                inputs["prompt"] = inputs["prompt"].replace(VIDEO_TOKEN, IMAGE_TOKEN * len(video_frames_inputs))
                if "image" not in inputs["multi_modal_data"]:
                    inputs["multi_modal_data"]["image"] = []
                inputs["multi_modal_data"]["image"].extend(video_frames_inputs)

                inputs["multi_modal_data"].pop("video", None)

        return inputs

    def judge_negative(text):
        is_negative = text.startswith('☟')
        return is_negative
    
    def get_first_special_token(text):
        """获取文本开头的特殊token"""
        text = text.strip()
        for token in SPECIAL_TOKEN_LIST:
            if text.startswith(token):
                return token
        return None
    
    def extract_instruction(text):
        """从文本中提取[INSTRUCTION]后面的内容"""
        if INSTRUCTION_TOKEN in text:
            parts = text.split(INSTRUCTION_TOKEN)
            if len(parts) > 1:
                return parts[-1].strip()
        return None

    async def stream_results(results_generator) -> AsyncGenerator[bytes, None]:
        previous_text = ""
        async for request_output in results_generator:

            text = request_output.outputs[0].text
            newly_generated_text = text[len(previous_text):]
            previous_text = text
            yield newly_generated_text

    async def collect_results_demo(results_generator):
        async for newly_generated_text in stream_results(results_generator):
            continue

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    worker_ready.set()
    if not isinstance(wait_workers_ready, list):
        wait_workers_ready = [wait_workers_ready]

    # 局部变量用于记录上一次的对话和视频帧
    last_dialog, last_video_frame_timestamp = None, 0.0
    action_generation_start_time = None
    action_generation_timeout = 20  # seconds

    flag = False
    while not shutdown_event.is_set():
        time.sleep(0.01)
        
        # Wait for all workers to be ready
        if not all([worker.is_set() for worker in wait_workers_ready]):
            time.sleep(0.1)
            continue

        if not flag:
            print(f"{Colors.CYAN}Process {cuda_devices} is ready.{Colors.RESET}")
            flag = True

        # 语音生成阶段
        if not inputs_queue.empty() and not is_generating_action_feature.is_set():
            # 有输入时，且没有在生成动作特征
            with start_event_lock:
                if start_event.is_set():
                    # start_event为True，表示允许生成
                    inputs = inputs_queue.get()
                    if not other_is_generating_action_feature.is_set():
                        # 另一个模型没有在生成动作特征，则下次让对方生成，自己不生成
                        other_start_event.set()
                        start_event.clear()
                else:
                    # 自己不被允许生成
                    continue
            
            # 处理输入
            inputs = _process_inputs(inputs)
            current_inputs = inputs.copy()
            inputs = merge_current_and_history(
                global_history[-global_history_limit:],
                inputs,
                skip_history_vision=True,
                move_image_token_to_start=True
            )
        
            if "prompt" in inputs:
                # Process multimodal tokens
                inputs["prompt_token_ids"] = tokenizer_image_audio_token(inputs["prompt"], tokenizer, image_token_index=IMAGE_TOKEN_INDEX, audio_token_index=AUDIO_TOKEN_INDEX)
            else:
                assert "prompt_token_ids" in inputs, "Either 'prompt' or 'prompt_token_ids' must be provided."

            print(f"{Colors.BLUE}Process {cuda_devices} is processing inputs: {inputs}{Colors.RESET}")
            inputs.pop("prompt", None)

            # 生成输出
            llm_start_time = time.time()
            output = llm.generate(inputs, sampling_params=sampling_params)
            llm_end_time = time.time()
            print(f"{Colors.GREEN}LLM process time: {llm_end_time - llm_start_time}{Colors.RESET}")

            llm_output = output[0].outputs[0].text
            print(f"{Colors.GREEN}LLM output: {llm_output}{Colors.RESET}")
            
            # 获取首个特殊token
            first_token = get_first_special_token(llm_output)
            print(f"{Colors.GREEN}First special token: {first_token}{Colors.RESET}")
            
            # 处理首token逻辑
            is_first_time_to_work = True
            if is_first_time_to_work:  # 检查第一个token的情况
                stop_event.clear()  # 允许把自己的输出发给下游，不被打断
                if not other_is_generating_action_feature.is_set():
                    other_stop_event.set()  # 如果对方没有在生成动作特征，可能在输出语音，则打断对方（语音打断）
                
                if first_token == HALT_TOKEN:
                    other_stop_event.set()  # 如果首 token 是急停，则打断对方（急停打断）
                    other_is_generating_action_feature.clear()  # 停止对方的动作特征生成
                    action_feature_queue.put({"id": llm_id, "action": HALT_TOKEN})  # 输出急停信号
                    
                elif first_token == ACT_TOKEN:
                    is_generating_action_feature.set()  # 如果首 token 是要输出动作，则开始生成动作特征
                    if not other_is_generating_action_feature.is_set():
                        while len(action_feature_stack) > 0:
                            action_feature_stack.pop() # 清空栈
                    action_generation_start_time = time.time() # 开始计时
                    # 通知前端或客户端开始更新机器人状态
                    print(f"{Colors.YELLOW}Process {cuda_devices} is notifying frontend to start updating robot states.{Colors.RESET}")
                    action_feature_queue.put({"id": llm_id, "action": "START_ACTION"})
                    try:
                        instruction_content = extract_instruction(llm_output)
                        if instruction_content:
                            current_dialog.value = instruction_content  # 获取动作指令，并赋值给共享变量
                        else:
                            warnings.warn("No [INSTRUCTION] found in outputs when [ACT] is found")
                            current_dialog.value = current_inputs.get("prompt", "")
                    except Exception as e:
                        warnings.warn(f"Error extracting instruction: {e}")
                        current_dialog.value = current_inputs.get("prompt", "")
                
                clear_queue(outputs_queue)  # 清空输出队列，准备输出自己的 token
                clear_queue(tts_outputs_queue)
                is_first_time_to_work = False

            # 处理输出到TTS队列（除了[INSTRUCTION]后面的部分）
            if not stop_event.is_set():
                # 分割输出，只发送[INSTRUCTION]之前的部分到TTS
                if INSTRUCTION_TOKEN in llm_output:
                    tts_output = llm_output.split(INSTRUCTION_TOKEN)[0]
                else:
                    tts_output = llm_output
                
                # 添加首句标记并发送到TTS
                tts_output = '$$FIRST_SENTENCE_MARK$$' + tts_output
                
                # 按字符处理TTS输出
                history_generated_text = ''
                for char in tts_output:
                    if stop_event.is_set():
                        print(f"{Colors.RED}Process {cuda_devices} is interrupted.{Colors.RESET}")
                        break
                    
                    history_generated_text += char
                    history_generated_text = history_generated_text.replace('☞ ', '').replace('☞', '')
                    
                    if char in [",", "，", ".", "。", "?", "\n", "？", "!", "！", "、"]:
                        outputs_queue.put({"id": llm_id, "response": history_generated_text})
                        history_generated_text = ''

            # 保存对话历史
            current_inputs["response"] = llm_output
            if llm_output.strip():
                global_history.append(current_inputs)

        # 动作特征生成阶段
        elif is_generating_action_feature.is_set():
            if other_is_generating_action_feature.is_set():
                print(f"{Colors.RED}Process {cuda_devices} is interrupting other action feature generation.{Colors.RESET}")
                other_stop_event.set()  # 如果对方在生成动作特征，则打断对方（动作打断）
                other_is_generating_action_feature.clear()
                other_start_event.set()  # 打断对方以后，下次让对方生成，自己不生成，自己进入动作特征生成阶段
                start_event.clear()  # 自己下次不被允许生成
            
            if stop_event.is_set():  # [TAG1] 检查是否被打断
                print(f"{Colors.RED}Process {cuda_devices} is interrupted.{Colors.RESET}")
                is_generating_action_feature.clear()  # 如果被打断，则停止生成动作特征
                stop_event.clear()  # 如果被打断，则下次允许生成自己的 token
                action_feature_queue.put({"id": llm_id, "action": HALT_TOKEN})  # 输出急停
                continue

            if len(action_feature_stack) > 0:  # 如果栈不为空，则执行回撤
                retraction_action = action_feature_stack.pop()
                action_feature_queue.put({"id": llm_id, "action": retraction_action})
                print(f"{Colors.YELLOW}Executing retraction action.{Colors.RESET}")
                continue

            # [TODO] 检查是否应该结束动作生成（这里需要根据实际模型实现来判断）
            # [TODO] 暂时用简单的逻辑，实际应该检查模型输出的[END] token
            # [TODO] 暂时先检查动作生成是否超时
            if action_generation_start_time and (time.time() - action_generation_start_time > action_generation_timeout):
                print(f"{Colors.RED}Process {cuda_devices} action generation timed out after {action_generation_timeout} seconds.{Colors.RESET}")
                is_generating_action_feature.clear()
                action_generation_start_time = None
                action_feature_queue.put({"id": llm_id, "action": HALT_TOKEN}) # 发送停止信号
                continue
            
            if is_generating_action_feature.is_set():  # 如果还在生成动作特征，没有被打断
                # 检查是否有新的对话或视频帧
                current_dialog_text = current_dialog.get("value", "")
                current_observation = None
                with observation_queue_lock:
                    if not observation_queue.empty():
                        current_observation = observation_queue.get()
                    else:
                        continue

                if current_observation is not None:
                    # 构建动作特征生成的输入
                    print(f"{Colors.YELLOW}Process {cuda_devices} is generating action feature.{Colors.RESET}")
                    dialog_text = current_dialog_text or ""
                    frame_data = current_observation.get("data", None)
                    robot_states = current_observation.get("states", {})  # 机器人状态

                    video_frame_array = None
                    if frame_data is not None:
                        video_frame_array = pickle.loads(frame_data)
                    
                    if video_frame_array is not None:
                        # 构建VITA输入（图片+指令，不附带历史消息）
                        action_inputs = {
                            "prompt": f"<image>{dialog_text}",
                            "multi_modal_data": {"image": [video_frame_array]}
                        }
                        
                        # 处理VITA输入
                        processed_action_inputs = _process_inputs(action_inputs)
                        if "prompt" in processed_action_inputs:
                            processed_action_inputs["prompt_token_ids"] = tokenizer_image_audio_token(
                                processed_action_inputs["prompt"], tokenizer, 
                                image_token_index=IMAGE_TOKEN_INDEX, audio_token_index=AUDIO_TOKEN_INDEX)
                        processed_action_inputs.pop("prompt", None)
                        
                        # 使用带hidden states提取的采样参数
                        sampling_params_with_hidden = SamplingParams(
                            temperature=0.01, 
                            max_tokens=1,  # 只需要获取hidden states，不需要生成文本
                            best_of=1, 
                            skip_special_tokens=False,
                            prefill_hidden_layer=27  # 提取第27层hidden states
                        )
                        
                        # 生成并获取hidden states
                        vita_output = llm.generate(processed_action_inputs, sampling_params=sampling_params_with_hidden)
                        
                        # 提取hidden states
                        hidden_states = vita_output[0].outputs[0].prefill_hidden_states
                        print(f"{Colors.YELLOW}Successfully extracted hidden states: {hidden_states.shape}{Colors.RESET}")
                        
                        # 使用GR00T action policy生成动作
                        if hidden_states is not None:
                            robot_observations = {
                                "state.hand": np.array(robot_states.get("hand"), dtype=np.float32)[None, :],
                                "state.robot": np.array(robot_states.get("robot"), dtype=np.float32)[None, :],
                            }
                            action_dict = gr00t_action_policy.get_action(hidden_states, observations=robot_observations)
                            
                            # 将CUDA Tensor转换为Python列表以便跨进程传递和JSON序列化
                            serializable_action_dict = {}
                            for k, v in action_dict.items():
                                if isinstance(v, torch.Tensor):
                                    serializable_action_dict[k] = v.cpu().numpy().tolist()
                                elif isinstance(v, np.ndarray):
                                    serializable_action_dict[k] = v.tolist()
                                else:
                                    serializable_action_dict[k] = v
                            
                            action_feature_stack.append(serializable_action_dict) # 将动作特征压入栈
                            action_feature_queue.put({"id": llm_id, "action": serializable_action_dict})
                            print(f"{Colors.YELLOW}Put action into action_feature_queue: {list(action_dict.keys())}{Colors.RESET}")
                        
                    else:
                        warnings.warn("No video frame found, cannot generate action")
                        is_generating_action_feature.clear()
                        stop_event.clear()
            else:  # 如果被打断，则输出急停
                action_feature_queue.put({"id": llm_id, "action": HALT_TOKEN})

def tts_worker(
    model_path,
    inputs_queue,
    outputs_queue,
    worker_ready,
    wait_workers_ready,
    shutdown_event,
):
    # 导入依赖CUDA的包
    import torch
    import torchaudio
    from vita.model.vita_tts.decoder.llm2tts import llm2TTS
    from vita.model.language_model.vita_qwen2 import VITAQwen2Config, VITAQwen2ForCausalLM
    from transformers import AutoTokenizer
    
    def audio_file_to_html(audio_file: str) -> str:
        """
        Convert audio file to HTML audio player.

        Args:
            audio_file: Path to audio file

        Returns:
            audio_player: HTML audio player that auto-plays
        """
        # Read in audio file to audio_bytes
        audio_bytes = io.BytesIO()
        with open(audio_file, "rb") as f:
            audio_bytes.write(f.read())

        # Generate audio player HTML object for autoplay
        audio_bytes.seek(0)
        audio = base64.b64encode(audio_bytes.read()).decode("utf-8")
        audio_player = (
            f'<audio src="data:audio/mpeg;base64,{audio}" controls autoplay></audio>'
        )
        return audio_player


    def remove_uncommon_punctuation(text):
        common_punctuation = ".,!?;:()[]，。！？、：；（） "
        uncommon_punctuation_pattern = rf"[^\w\s{re.escape(common_punctuation)}]"
        cleaned_text = re.sub(uncommon_punctuation_pattern, "", text)

        return cleaned_text
    
    def remove_special_tokens(input_str):
        # Remove special tokens
        special_tokens = ['☞', '☟', '☜', '☝', '☀', '☯', '<unk>', '<|im_end|>']
        for token in special_tokens:
            input_str = input_str.replace(token, '')
        return input_str

    def replace_equation(sentence):

        special_notations = {
            "sin": " sine ",
            "cos": " cosine ",
            "tan": " tangent ",
            "cot": " cotangent ",
            "sec": " secant ",
            "csc": " cosecant ",
            "log": " logarithm ",
            "exp": "e^",
            "sqrt": "根号 ",
            "abs": "绝对值 ",
        }
        
        special_operators = {
            "+": "加",
            "-": "减",
            "*": "乘",
            "/": "除",
            "=": "等于",
            '!=': '不等于',
            '>': '大于',
            '<': '小于',
            '>=': '大于等于',
            '<=': '小于等于',
        }

        greek_letters = {
            "α": "alpha ",
            "β": "beta ",
            "γ": "gamma ",
            "δ": "delta ",
            "ε": "epsilon ",
            "ζ": "zeta ",
            "η": "eta ",
            "θ": "theta ",
            "ι": "iota ",
            "κ": "kappa ",
            "λ": "lambda ",
            "μ": "mu ",
            "ν": "nu ",
            "ξ": "xi ",
            "ο": "omicron ",
            "π": "派 ",
            "ρ": "rho ",
            "σ": "sigma ",
            "τ": "tau ",
            "υ": "upsilon ",
            "φ": "phi ",
            "χ": "chi ",
            "ψ": "psi ",
            "ω": "omega "
        }

        sentence = sentence.replace('**', ' ')

        sentence = re.sub(r'(?<![\d)])-(\d+)', r'负\1', sentence)

        for key in special_notations:
            sentence = sentence.replace(key, special_notations[key]) 
        for key in special_operators:
            sentence = sentence.replace(key, special_operators[key])
        for key in greek_letters:
            sentence = sentence.replace(key, greek_letters[key])


        sentence = re.sub(r'\(?(\d+)\)?\((\d+)\)', r'\1乘\2', sentence)
        sentence = re.sub(r'\(?(\w+)\)?\^\(?(\w+)\)?', r'\1的\2次方', sentence)
        
        return sentence

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_embedding = load_model_embemding(model_path).to(device)
    tts = llm2TTS(os.path.join(model_path, 'vita_tts_ckpt/'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


    worker_ready.set()
    if not isinstance(wait_workers_ready, list):
        wait_workers_ready = [wait_workers_ready]

    past_llm_id = 0

    while not shutdown_event.is_set():
        # Wait for all workers to be ready
        if not all([worker.is_set() for worker in wait_workers_ready]):
            time.sleep(0.1)
            continue

        tts_input_text = ""
        while not inputs_queue.empty():
            time.sleep(0.03)

            stop_at_punc_or_len = False
            response = inputs_queue.get()
            llm_id, newly_generated_text = response["id"], response["response"]

            for character in newly_generated_text:
                
                if  past_llm_id != 0 and past_llm_id != llm_id:
                    tts_input_text = ""
                    outputs_queue.put({"id": llm_id, "response": ("|PAUSE|", None, 0.2)})
                
                tts_input_text += character

                past_llm_id = llm_id
                if character in [",", "，", ".", "。", "?", "\n", "？", "!", "！", "、"] and len(tts_input_text) >= 5:
                    stop_at_punc_or_len = True
                    break

            if stop_at_punc_or_len:
                break

        if tts_input_text.strip() == "":
            continue

        if '$$FIRST_SENTENCE_MARK$$' in  tts_input_text.strip():
            codec_chunk_size = 20
            seg_threshold = 0.1
            tts_input_text = tts_input_text.replace('$$FIRST_SENTENCE_MARK$$', '').replace('，', '。').replace(',', '。')
            IS_FIRST_SENTENCE = True
        else:
            codec_chunk_size = 40
            seg_threshold = 0.015
            IS_FIRST_SENTENCE = False
        tts_input_text = remove_special_tokens(tts_input_text)
        tts_input_text = replace_equation(tts_input_text)
        tts_input_text = tts_input_text.lower()

        if tts_input_text.strip() == "":
            continue
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embeddings = llm_embedding(torch.tensor(tokenizer.encode(tts_input_text)).to(device))
        for seg in tts.run(embeddings.reshape(-1, 896).unsqueeze(0), decoder_topk,
                            None, 
                            codec_chunk_size=codec_chunk_size,
                            codec_padding_size=codec_padding_size,
                            seg_threshold=seg_threshold):

            if IS_FIRST_SENTENCE:
                try:
                    split_idx = torch.nonzero(seg.abs() > 0.03, as_tuple=True)[-1][0]
                    seg = seg[:, :, split_idx:]
                except:
                    print(f'{Colors.MAGENTA}Do not need to split{Colors.RESET}')
                    pass

            seg = torch.cat([seg], -1).float().cpu()
            audio_data = (seg.squeeze().numpy() * 32768.0).astype(np.int16)

            audio_duration = seg.shape[-1]/24000
            if past_llm_id == 0 or past_llm_id == llm_id:
                outputs_queue.put({"id": llm_id, "response": (tts_input_text, audio_data, audio_duration)})

def send_pcm(sid, request_inputs_queue):
    """
    Sends PCM audio data to the dialogue system for processing.
    """
    chunk_size = connected_users[sid][1].wakeup_and_vad.get_chunk_size()

    print(f"Sid: {sid} Start listening")
    while True:
        if connected_users[sid][1].stop_pcm:
            print(f"Sid: {sid} Stop pcm")
            connected_users[sid][1].stop_generate = True 
            connected_users[sid][1].stop_tts = True
            break
            
        time.sleep(0.01)
        e = connected_users[sid][1].pcm_fifo_queue.get(chunk_size)
        if e is None:
            continue

        res = connected_users[sid][1].wakeup_and_vad.predict(e)

        if res is not None:
            if 'start' in res:
                print(f"Sid: {sid} Vad start")

            elif 'cache_dialog' in res:
                print(f"Sid: {sid} Vad end")

                directory = './chat_history'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                audio_duration = len(res["cache_dialog"]) / target_sample_rate

                if audio_duration < 1:
                    print(f"{Colors.YELLOW}The duration of the audio is less than 1s, skipping...{Colors.RESET}")
                    continue

                current_time = datetime.datetime.now()
                timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                audio_filename = f"{directory}/test_dialog_{timestamp}.wav"
                torchaudio.save(audio_filename, res["cache_dialog"].unsqueeze(0), target_sample_rate)

                video_filename = None
                if len(connected_users[sid][1].collected_images) > 0:
                    video_filename = f"{directory}/test_video_{timestamp}.mp4"
                    save_video(connected_users[sid][1].collected_images, video_filename)

                print(f"{Colors.BLUE}Start to generate response{Colors.RESET}")
                if video_filename:
                    current_request = {
                        "prompt": "<video><audio>",
                        "multi_modal_data": {
                            "video": [video_filename],
                            "audio": [audio_filename],
                        },
                    }
                else:
                    current_request = {
                        "prompt": "<audio>",
                        "multi_modal_data": {
                            "audio": [audio_filename],
                        },
                    }
                print(f"{Colors.BLUE}Start to put request into queue {current_request}{Colors.RESET}")
                request_inputs_queue.put(current_request)

@app.route('/')
def index():
    return render_template('demo.html')

@socketio.on('connect')
def handle_connect():
    if len(connected_users) >= args.max_users:
        print(f'{Colors.YELLOW}Too many users connected, disconnecting new user{Colors.RESET}')
        emit('too_many_users')
        return

    sid = request.sid
    connected_users[sid] = []
    connected_users[sid].append(Timer(args.timeout, disconnect_user, [sid]))
    connected_users[sid].append(GlobalParams())
    connected_users[sid][0].start()
    
    request_queue = current_app.config['REQUEST_QUEUE']
    pcm_thread = threading.Thread(target=send_pcm, args=(sid, request_queue,))
    pcm_thread.start()
    print(f'{Colors.CYAN}User {sid} connected{Colors.RESET}')

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][0].cancel()
        connected_users[sid][1].interrupt()
        connected_users[sid][1].stop_pcm = True
        connected_users[sid][1].release()
        time.sleep(3)
        del connected_users[sid]
    print(f'{Colors.CYAN}User {sid} disconnected{Colors.RESET}')

@socketio.on('recording-started')
def handle_recording_started():
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][0].cancel()
        connected_users[sid][0] = Timer(args.timeout, disconnect_user, [sid])
        connected_users[sid][0].start()
        connected_users[sid][1].interrupt()
        socketio.emit('stop_tts', to=sid)
        connected_users[sid][1].reset()
    else:
        disconnect()
    print('Recording started')

@socketio.on('recording-stopped')
def handle_recording_stopped():
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][0].cancel()
        connected_users[sid][0] = Timer(args.timeout, disconnect_user, [sid])
        connected_users[sid][0].start()
        connected_users[sid][1].interrupt()
        socketio.emit('stop_tts', to=sid)
        connected_users[sid][1].reset()
    else:
        disconnect()
    print('Recording stopped')

@socketio.on('audio')
def handle_audio(data):
    global last_tts_model_id
    sid = request.sid
    if sid in connected_users:
        try:
            # 处理action_feature队列中的数据
            action_feature_queue = current_app.config.get('ACTION_FEATURE_QUEUE')
            action_feature_stack = current_app.config.get('ACTION_FEATURE_STACK')

            if action_feature_queue and not action_feature_queue.empty():
                try:
                    action_data = action_feature_queue.get_nowait()
                    action = action_data.get('action')
                    print(f"{Colors.BLUE}Processing action queue: {action_data.keys()}{Colors.RESET}")

                    if action == "START_ACTION":
                        print(f"{Colors.YELLOW}Received START_ACTION signal, emitting start_update_states.{Colors.RESET}")
                        emit('start_update_states', True)
                    elif isinstance(action, dict):
                        # This is an actual action feature
                        print(f"{Colors.YELLOW}Sending action feature to frontend: {list(action.keys())}{Colors.RESET}")
                        emit('action_feature', action)
                    elif action == HALT_TOKEN:
                        print(f"{Colors.RED}Sending HALT signal to frontend.{Colors.RESET}")
                        emit('action_feature', {"halt": True})
                except Empty:
                    pass
            
            if not current_app.config['TTS_OUTPUT_QUEUE'].empty():
                connected_users[sid][0].cancel()
                connected_users[sid][0] = Timer(args.timeout, disconnect_user, [sid])
                connected_users[sid][0].start()

                tts_output_queue = current_app.config['TTS_OUTPUT_QUEUE']
                try:
                    output_data = tts_output_queue.get_nowait()
                    print(f"{Colors.MAGENTA}output_data: {output_data}{Colors.RESET}")

                    if output_data is not None:
                        llm_id = output_data["id"]
                        _, audio, length = output_data["response"]

                        print(f"{Colors.MAGENTA}llm_id: {llm_id}, last_tts_model_id: {last_tts_model_id}{Colors.RESET}")
                        if last_tts_model_id != llm_id:
                            print(f"{Colors.YELLOW}Received output from other process {llm_id}, last output tts model is {last_tts_model_id}, skipping...{Colors.RESET}")
                            socketio.emit('stop_tts', to=sid)
                        else:
                            print(f"Sid: {sid} Send TTS data")
                            emit('audio', audio.tobytes())

                        last_tts_model_id = llm_id
                except Empty:
                    pass
        
            if connected_users[sid][1].tts_over_time > 0:
                socketio.emit('stop_tts', to=sid)
                connected_users[sid][1].tts_over_time = 0
            
            data = json.loads(data)
            audio_data = np.frombuffer(bytes(data['audio']), dtype=np.int16)
            sample_rate = data['sample_rate']
            
            connected_users[sid][1].pcm_fifo_queue.put(torch.tensor(audio_data, dtype=torch.float32) / 32768.0)

        except Exception as e:
            traceback.print_exc()
            print(f"{Colors.RED}Error processing audio: {e}{Colors.RESET}")
    else:
        disconnect()

@socketio.on('video_frame')
def handle_video_frame(data):
    sid = request.sid
    if sid in connected_users:
        try:
            image_data = base64.b64decode(data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            current_time = time.time()
            if current_time - connected_users[sid][1].last_image_time > 1:
                connected_users[sid][1].collected_images.clear()
                print(f"{Colors.MAGENTA}Clearing the collected images{Colors.RESET}")
            
            connected_users[sid][1].collected_images.append(frame)
            connected_users[sid][1].last_image_time = current_time
            
        except Exception as e:
            print(f"{Colors.RED}Error processing video frame: {e}{Colors.RESET}")
    else:
        disconnect()

@socketio.on('reset_state')
def handle_reset_state():
    global_history = current_app.config['GLOBAL_HISTORY']
    while len(global_history) > 0:
        global_history.pop()
    print(f"{Colors.CYAN}Resetting the state{Colors.RESET}")

@socketio.on('states')
def handle_states(data):
    observation_queue = current_app.config.get('OBSERVATION_QUEUE')
    observation_queue_lock = current_app.config.get('OBSERVATION_QUEUE_LOCK')
    with observation_queue_lock:
        while not observation_queue.empty():
            observation_queue.get()
        
        # 处理图像数据，与video_frame处理方式统一
        processed_data = data.copy()
        if data.get('data') is not None:
            try:
                # 解码base64图像数据（与handle_video_frame相同的处理方式）
                image_data = base64.b64decode(data['data'].split(',')[1])
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 使用pickle序列化，与原来的处理方式保持一致
                processed_data['data'] = pickle.dumps(frame)
                print(f"{Colors.BLUE}Processed image data, frame shape: {frame.shape}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.RED}Error processing image data: {e}{Colors.RESET}")
                processed_data['data'] = None
        
        # data格式：{"data": pickle.dumps(frame) or None, "states": {"hand": [...], "robot": [...]}}
        observation_queue.put(processed_data)
        print(f"{Colors.BLUE}Added robot states to queue: hand={len(data.get('states', {}).get('hand', []))}, robot={len(data.get('states', {}).get('robot', []))}{Colors.RESET}")

def cleanup_resources():
    """清理多进程资源"""
    print(f"{Colors.CYAN}正在清理资源...{Colors.RESET}")
    with app.app_context():
        # 1. 通知所有子进程关闭
        if 'SHUTDOWN_EVENT' in current_app.config:
            print(f"{Colors.CYAN}发送关闭信号...{Colors.RESET}")
            current_app.config['SHUTDOWN_EVENT'].set()

        # 2. 等待进程结束
        processes_to_join = {
            'MODEL_1_PROCESS': current_app.config.get('MODEL_1_PROCESS'),
            'MODEL_2_PROCESS': current_app.config.get('MODEL_2_PROCESS'),
            'TTS_WORKER_PROCESS': current_app.config.get('TTS_WORKER_PROCESS'),
        }

        for name, process in processes_to_join.items():
            if process and process.is_alive():
                print(f"{Colors.CYAN}等待 {name} 退出...{Colors.RESET}")
                process.join(timeout=10) # 等待10秒

        # 3. 清空队列
        queues_to_clear = [
            current_app.config.get('REQUEST_QUEUE'),
            current_app.config.get('TTS_QUEUE'),
            current_app.config.get('TTS_OUTPUT_QUEUE'),
            current_app.config.get('ACTION_FEATURE_QUEUE'),
            current_app.config.get('OBSERVATION_QUEUE'),
        ]
        print(f"{Colors.CYAN}清空队列...{Colors.RESET}")
        for queue in queues_to_clear:
            if queue:
                clear_queue(queue)
        
        # 4. 强制终止仍在运行的进程
        for name, process in processes_to_join.items():
            if process and process.is_alive():
                print(f"{Colors.RED}进程 {name} 未能正常退出，强制终止。{Colors.RESET}")
                process.terminate()
                process.join()

    print(f"{Colors.CYAN}资源清理完成。{Colors.RESET}")

if __name__ == "__main__":
    print(f"{Colors.CYAN}Start VITA server{Colors.RESET}")
    
    # 1. 初始化多进程相关资源
    multiprocessing.set_start_method('spawn', force=True)
    manager = multiprocessing.Manager()
    
    # 使用ProcessConfig类简化配置管理
    config = ProcessConfig(args, manager)

    # 2. 启动工作进程
    tts_worker_process = multiprocessing.Process(
        target=tts_worker,
        kwargs=config.get_tts_worker_kwargs()
    )

    model_1_process = multiprocessing.Process(
        target=load_model,
        kwargs=config.get_model_1_kwargs()
    )

    model_2_process = multiprocessing.Process(
        target=load_model,
        kwargs=config.get_model_2_kwargs()
    )

    # 3. 启动进程
    model_1_process.start()
    model_2_process.start()
    tts_worker_process.start()

    # 4. 将多进程资源添加到 Flask app context
    app_config = config.get_app_config()
    app_config.update({
        'MODEL_1_PROCESS': model_1_process,
        'MODEL_2_PROCESS': model_2_process,
        'TTS_WORKER_PROCESS': tts_worker_process,
    })
    
    for key, value in app_config.items():
        app.config[key] = value

    import cv2
    import pickle
    import torch
    import torchaudio
    
    # 5. 启动 Flask 应用
    cert_file = "web_demo/vita_html/web/resources/cert.pem"
    key_file = "web_demo/vita_html/web/resources/key.pem"
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        generate_self_signed_cert(cert_file, key_file)
    
    try:
        socketio.run(app, host=args.ip, port=args.port, debug=False, ssl_context=(cert_file, key_file))
    finally:
        print(f"{Colors.CYAN}捕获到退出信号，开始清理...{Colors.RESET}")
        cleanup_resources()

    # 6. 等待进程结束
    model_1_process.join()
    model_2_process.join()
    tts_worker_process.join()
