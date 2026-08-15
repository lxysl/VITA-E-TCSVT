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
import warnings

import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy_vita import BasePolicy, Gr00tPolicy
from gr00t.utils.eval_vita import calc_mse_for_single_trajectory

warnings.simplefilter("ignore", category=FutureWarning)

"""
Example command:

python scripts/eval_policy.py --host localhost --port 5555 --plot
    --modality_keys right_arm right_hand
    --steps 250
    --trajs 1000
    --exec_horizon 16
    --video_backend decord
    --dataset_path demo_data/robot_sim.PickNPlace/
    --embodiment_tag gr1
    --data_config gr1_arms_waist
provide --model_path to load up the model checkpoint in this script.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost", help="host")
    parser.add_argument("--port", type=int, default=8001, help="port")
    parser.add_argument("--plot", action="store_true", help="plot images")
    parser.add_argument("--modality_keys", nargs="+", type=str, default=["state"])
    parser.add_argument(
        "--data_config",
        type=str,
        default="libero_vita",
        choices=list(DATA_CONFIG_MAP.keys()),
        help="data config name",
    )
    parser.add_argument("--steps", type=int, default=150, help="number of steps to run")
    parser.add_argument("--trajs", type=int, default=1, help="trajectories to run")
    parser.add_argument("--exec_horizon", type=int, default=8)
    parser.add_argument("--video_backend", type=str, default="torchvision_av")
    parser.add_argument("--dataset_path", type=str, default="/root/Isaac-GR00T/dataset/libero_90_no_noops_lerobot_fixed")
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="new_embodiment",
    )
    ## When using a model instead of client-server mode.
    parser.add_argument(
        "--model_path",
        type=str,
        # default=None,
        # default="/root/Isaac-GR00T/checkpoints/libero_90_debug1/checkpoint-2",
        # default="/root/Isaac-GR00T/checkpoints/vita_libero_spatial_bs64_gpu8/checkpoint-34000",
        # default="/root/Isaac-GR00T/checkpoints/vita_libero_spatial_bs32_gpu8_tune_llm/checkpoint-24000",
        # default="/root/Isaac-GR00T/checkpoints/vita_libero_90_bs32_gpu8_tune_llm/checkpoint-47000",
        # default="/root/Isaac-GR00T/checkpoints/vita_libero_90_fixed_bs64_gpu8_frozen_vlm/checkpoint-8000",
        default="/root/Isaac-GR00T/checkpoints/vita_libero_90_fixed_bs64_gpu8_frozen_vlm_fix_lang/checkpoint-21000",
        # default="/root/Isaac-GR00T/checkpoints/vita_libero_spatial_bs8_gpu8_tune_visual_fixed/checkpoint-80000",
        # default="/root/Isaac-GR00T/checkpoints/vita_libero_10_bs8_gpu8_tune_visual_fixed/checkpoint-75000",
        # default="/root/Isaac-GR00T/checkpoints/vita_libero_spatial_bs64_gpu8_frozen_vlm/checkpoint-13000",
        help="[Optional] Path to the model checkpoint directory, this will disable client server mode.",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        help="Number of denoising steps if model_path is provided",
        default=4,
    )
    args = parser.parse_args()

    data_config = DATA_CONFIG_MAP[args.data_config]
    if args.model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy: BasePolicy = RobotInferenceClient(host=args.host, port=args.port)

    all_gt_actions = []
    all_pred_actions = []

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print(modality)

    # Create the dataset
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=args.embodiment_tag,
    )

    print(len(dataset))
    # Make a prediction
    obs = dataset[0]
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    for k, v in dataset.get_step_data(0, 0).items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    print("Total trajectories:", len(dataset.trajectory_lengths))
    print("All trajectories:", dataset.trajectory_lengths)
    print("Running on all trajs with modality keys:", args.modality_keys)

    all_mse = []
    for traj_id in range(args.trajs):
        print("Running trajectory:", traj_id)
        mse = calc_mse_for_single_trajectory(
            policy,
            dataset,
            traj_id,
            modality_keys=args.modality_keys,
            steps=args.steps,
            exec_horizon=args.exec_horizon,
            plot=args.plot,
            realtime_flag=True
        )
        print("MSE:", mse)
        all_mse.append(mse)
    print("Average MSE across all trajs:", np.mean(all_mse))
    print("Done")
    exit()
