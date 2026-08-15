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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy_vita import BasePolicy

# numpy print precision settings 3, dont use exponential notation
np.set_printoptions(precision=3, suppress=True)


def download_from_hg(repo_id: str, repo_type: str) -> str:
    """
    Download the model/dataset from the hugging face hub.
    return the path to the downloaded
    """
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download(repo_id, repo_type=repo_type)
    return repo_path


def calc_mse_for_single_trajectory(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    steps=300,
    exec_horizon=16,
    plot=False,
    realtime_flag=False
):
    state_joints_across_time = []
    gt_action_joints_across_time = []
    pred_action_joints_across_time = []
    pred_action_origin_joints_across_time = []
    if realtime_flag:
        print("use real-time chunking")
        prev_action_chunk = None
    
    for step_count in range(steps):
        data_point = dataset.get_step_data(traj_id, step_count)

        # NOTE this is to get all modality keys concatenated
        # concat_state = data_point["state.state"][0]
        # concat_gt_action = data_point["action.actions"][0]
        # concat_state = np.concatenate(
        #     [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
        # )
        # modality_keys=["actions"]
        # concat_gt_action = np.concatenate(
        #     [data_point[f"action.{key}"][0] for key in modality_keys], axis=0
        # )
        concat_state = np.concatenate(
            [data_point["state.state"][0][:7]], axis=0
        )
        concat_gt_action = np.concatenate(
            [data_point["action.actions"][0]], axis=0
        )
        if realtime_flag:
            data_point["prev_action_chunk"]=prev_action_chunk
            data_point["inference_delay"]=4

        state_joints_across_time.append(concat_state)
        gt_action_joints_across_time.append(concat_gt_action)

        if step_count % exec_horizon == 0:
            print("inferencing at step: ", step_count)
            if realtime_flag:
                prev_ = tmp_test
                realtime_action = policy.get_realtime_action(data_point)
                action_chunk = realtime_action["action.actions"][:exec_horizon]
                prev_action_chunk = realtime_action["prev_action_chunk"]
                tmp_test = realtime_action["tmp_test"]
                
                action_chunk_origin = policy.get_action(data_point)["action.actions"][:exec_horizon]
            else:
                action_chunk = policy.get_action(data_point)["action.actions"]
            
            for j in range(exec_horizon):
                # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                # concat_pred_action = np.concatenate(
                #     [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                #     axis=0,
                # )
                concat_pred_action = np.concatenate(
                    [action_chunk[j]],
                    axis=0,
                )
                print(concat_pred_action)
                
                pred_action_joints_across_time.append(concat_pred_action)

                if realtime_flag:
                    concat_pred_action_origin = np.concatenate(
                        [action_chunk_origin[j]],
                        axis=0,
                    )
                    pred_action_origin_joints_across_time.append(concat_pred_action_origin)


    # plot the joints
    state_joints_across_time = np.array(state_joints_across_time)
    gt_action_joints_across_time = np.array(gt_action_joints_across_time)
    pred_action_joints_across_time = np.array(pred_action_joints_across_time)[:steps]
    if realtime_flag:
        pred_action_origin_joints_across_time = np.array(pred_action_origin_joints_across_time)[:steps]
        assert (pred_action_joints_across_time.shape==pred_action_origin_joints_across_time.shape)
    print(state_joints_across_time.shape, gt_action_joints_across_time.shape, pred_action_joints_across_time.shape)
    assert (
        state_joints_across_time.shape
        == gt_action_joints_across_time.shape
        == pred_action_joints_across_time.shape
    )

    # calc MSE across time
    mse = np.mean((gt_action_joints_across_time - pred_action_joints_across_time) ** 2)
    if realtime_flag:
        mse2 = np.mean((gt_action_joints_across_time - pred_action_origin_joints_across_time) ** 2)
        print("Unnormalized Action MSE across single traj->", "    real-time chunking mse: ", mse, "    origin mse: ", mse2)
    else:
        print("Unnormalized Action MSE across single traj:", mse)

    num_of_joints = state_joints_across_time.shape[1]

    if plot:
        fig, axes = plt.subplots(nrows=num_of_joints, ncols=1, figsize=(8, 4 * num_of_joints))

        # Add a global title showing the modality keys
        names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        fig.suptitle(
            # f"Trajectory {traj_id} - Modalities: {', '.join(modality_keys)}",
            f"Trajectory {traj_id} - Modalities: {', '.join(names)}",
            fontsize=16,
            color="blue",
        )

        for i, ax in enumerate(axes):
            # ax.plot(state_joints_across_time[:, i], label="state joints")
            ax.plot(gt_action_joints_across_time[:, i], label="gt action")
            ax.plot(pred_action_joints_across_time[:, i], label="pred action")
            if realtime_flag:
                ax.plot(pred_action_origin_joints_across_time[:, i], label="pred action origin")

            # put a dot every ACTION_HORIZON
            for j in range(0, steps, exec_horizon):
                if j == 0:
                    ax.plot(j, gt_action_joints_across_time[j, i], "ro", label="inference point")
                else:
                    ax.plot(j, gt_action_joints_across_time[j, i], "ro")

            ax.set_title(f"Action dim {i}")
            ax.legend()

        plt.tight_layout()
        plt.savefig(f"/root/Isaac-GR00T/plot_pic/traj_{traj_id}_joints.png")  # ±£´æÍ¼Æ¬
        plt.close(fig)

    return mse
