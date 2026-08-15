import os
import warnings

import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, logging

from vl_load.vita.constants import GLOBAL_WEIGHTS_PATH
from vl_load.vita.model import *

logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    model_type,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    output_hidden_states=False,
    load_act_tokenizer=False,
    **kwargs,
):
    if model_type not in {"mixtral-8x7b", "nemo", "qwen2p5_instruct", "qwen2p5_fo_instruct"}:
        raise ValueError(f"Unknown Model Type {model_type}")

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}


    kwargs["torch_dtype"] = torch.bfloat16

    # Load VITA model
    if True:
        if model_type == "qwen2p5_instruct":
            # import pdb; pdb.set_trace()
            if load_act_tokenizer:
                tokenizer = AutoTokenizer.from_pretrained(model_path+"/../../../../tokenizer_with_act", use_fast=True)
                print("Load act tokenizer!")
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = VITAQwen2ForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, output_hidden_states=output_hidden_states, **kwargs
            )

    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()

    num_params = sum(p.numel() for p in vision_tower.parameters())
    print("the number of vision encoder params: {}M".format(num_params / 1024 / 1024))

    if getattr(model.config, "unfreeze_vision_tower", False):
        if True:
            assert model_base is None
            from safetensors.torch import load_file

            vision_weights = {}
            for file_name in os.listdir(model_path):
                if file_name.endswith("safetensors"):
                    vision_weights.update(
                        {
                            k[19:]: v
                            for k, v in load_file(os.path.join(model_path, file_name)).items()
                            if k.startswith("model.vision_tower.")
                        }
                    )
            vision_tower.load_state_dict(vision_weights, strict=True)


    vision_tower.to(dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor

    #import pdb; pdb.set_trace()
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return tokenizer, model, image_processor, context_len