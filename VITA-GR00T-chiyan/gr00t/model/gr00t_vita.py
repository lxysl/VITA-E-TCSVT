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

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .backbone import EagleBackbone
import os

from .vita_model import VITAModel
# from .vita_model_act import VITAModel


BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class GR00T_N1Config(PretrainedConfig):
    model_type = "gr00t_n1"
    # backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00T_N1(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1Config,
        local_model_path: str,
    ):
        # assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        # self.backbone = EagleBackbone(**config.backbone_cfg)
        # self.vita_model = VITAModel(model_path="./gr00t/model/vl_load/checkpoints/VITA-1.5", p_num=[1])
        self.vita_model = VITAModel(model_path="/root/VITA/checkpoints/vita_vla_finetune_ended/llava-s3-finetune_task_neg", p_num=[1])
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype
        
    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        # print(inputs.keys())
        vita_inputs, action_inputs = self.prepare_input(inputs)
        # print("vita_inputs: ", vita_inputs.keys())
        # print("backbone_inputs: ", backbone_inputs.keys())
        # print("action_inputs: ", action_inputs.keys())
        # print("here1: ", backbone_inputs["state"].shape, backbone_inputs["state_mask"].shape, backbone_inputs["segmentation_target"].shape, backbone_inputs["segmentation_target_mask"].shape)
        # # torch.Size([8, 1, 64]) torch.Size([8, 1, 64]) torch.Size([8, 2]) torch.Size([8, 1])
        # print("here2: ", backbone_inputs["has_real_action"], backbone_inputs["action"].shape, backbone_inputs["action_mask"].shape)
        # # tensor([True, True, True, True, True, True, True, True], device='cuda:0') torch.Size([8, 16, 32]) torch.Size([8, 16, 32])
        # print("here3: ", backbone_inputs["embodiment_id"], backbone_inputs["pixel_values"].shape, backbone_inputs["input_ids"].shape, backbone_inputs["attention_mask"].shape)
        # # tensor([31, 31, 31, 31, 31, 31, 31, 31], device='cuda:0') torch.Size([16, 3, 224, 224]) torch.Size([8, 183]) torch.Size([8, 183])

        # backbone_outputs = self.backbone(backbone_inputs)
        backbone_outputs = self.vita_model(vita_inputs)
        
        # print("backbone_outputs: ", backbone_outputs["backbone_features"].shape, backbone_outputs["backbone_attention_mask"].shape) # seq length is not fixed, feature size 1536
        # # torch.Size([8, 183, 1536]) torch.Size([8, 183])
        
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        vita_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        # backbone_outputs = self.backbone(backbone_inputs)
        backbone_outputs = self.vita_model(vita_inputs)
        # print("backbone_outputs: ", backbone_outputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def get_realtime_action(
        self,
        inputs,
        prev_action_chunk,
        inference_delay,
        prefix_attention_horizon
    ) -> BatchFeature:
        vita_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        # backbone_outputs = self.backbone(backbone_inputs)
        backbone_outputs = self.vita_model(vita_inputs)
        # print("backbone_outputs: ", backbone_outputs)
        action_head_outputs = self.action_head.get_realtime_action(backbone_outputs, 
                                                                   action_inputs,
                                                                   prev_action_chunk,
                                                                   inference_delay,
                                                                   prefix_attention_horizon)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        # backbone_inputs = self.backbone.prepare_input(inputs)
        vita_inputs = self.vita_model.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)
        
        # print("action_head: ", list(self.action_head.parameters()))
        # print("action_head dtype: ", self.action_head.dtype)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        # backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        vita_inputs = tree.map_structure(to_device_with_maybe_dtype, vita_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        
        return vita_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", False)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", False)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", False)
        load_separately = kwargs.pop("load_separately", False)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        if False:
            # get the current model path being downloaded
            try:
                # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
                # saved in ~/.cache/huggingface/hub/
                local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
                # HFValidationError, RepositoryNotFoundError
            except (HFValidationError, RepositoryNotFoundError):
                print(
                    f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
                )
                local_model_path = pretrained_model_name_or_path
        else:
            print(
                f"Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        ) # ignore_mismatched_sizes=True,

        # for name, param in pretrained_model.named_parameters():
        #     print(f"{name}")
        #     if name == "backbone.model.vision_model.vision_model.encoder.layers.9.mlp.fc2.weight":
        #         print(f"{name}")
        #         print(f"{param}")
        #         print(f"{param.shape}")
        #         break
        
        if True:
            if hasattr(pretrained_model, "backbone"):
                del pretrained_model.backbone
        # pretrained_model.backbone.set_trainable_parameters(
        #     tune_visual=tune_visual, tune_llm=tune_llm
        # )
        
        # vita encoder
        print("Loading VITA model...")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_id = torch.device(f"cuda:{local_rank}")
        pretrained_model.vita_model.init_model(device_id=device_id, tune_visual=tune_visual, tune_llm=tune_llm, load_separately=load_separately)
        print("VITA model successed.")
        
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model


# register
AutoConfig.register("gr00t_n1", GR00T_N1Config)
AutoModel.register(GR00T_N1Config, GR00T_N1)
