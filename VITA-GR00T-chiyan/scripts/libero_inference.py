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

from gr00t.eval.robot import RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        # default="/root/Isaac-GR00T/checkpoints/libero_spatial_20000_modality_fixed/checkpoint-20000",
        # default="/root/Isaac-GR00T/checkpoints/libero_object_20000/checkpoint-20000",
        # default="/root/Isaac-GR00T/checkpoints/libero_goal_20000/checkpoint-20000",
        # default="/root/Isaac-GR00T/checkpoints/libero_10_20000/checkpoint-20000",
        # default="/root/Isaac-GR00T/checkpoints/libero_90_20000/checkpoint-46000",
        default="/root/Isaac-GR00T/checkpoints/gr00t_libero_90_bs64_gpu8/checkpoint-20000",
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
        default="libero_90", # libero_spatial
        # default="libero_object",
    )

    parser.add_argument("--port", type=int, help="Port number for the server.", default=8001)
    parser.add_argument(
        "--host", type=str, help="Host address for the server.", default="0.0.0.0"
    )
    parser.add_argument("--denoising_steps", type=int, help="Number of denoising steps.", default=4)
    args = parser.parse_args()

    # Create a policy
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
