
import base64
import os
from io import BytesIO
from typing import List, Union

import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoTokenizer, CLIPImageProcessor
from transformers.feature_extraction_utils import BatchFeature
import yaml
import json

import gr00t


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from vl_load.vita.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from vl_load.vita.util.mm_utils import tokenizer_image_token
from vl_load.vita.conversation import conv_templates

import warnings
from transformers import logging
from vl_load.vita.model import *
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

DEFAULT_VITA_MODEL_NAME = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "vl_load", "checkpoints", "VITA-1.5"
)

DEFAULT_TOKENIZER_MODEL_NAME = os.path.join(
    os.path.dirname(gr00t.__file__), "tokenizer_with_act"
)

DEFAULT_VIT_MODEL_NAME = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "vl_load", "checkpoints", "InternViT-300M-448px"
)

def load_image(image):
    if isinstance(image, str) and os.path.exists(image):
        return Image.open(image)
    elif isinstance(image, dict):
        if "disk_path" in image:
            return Image.open(image["disk_path"])
        elif "base64" in image:
            return Image.open(BytesIO(base64.b64decode(image["base64"])))
        elif "url" in image:
            response = requests.get(image["url"])
            return Image.open(BytesIO(response.content))
        elif "bytes" in image:
            return Image.open(BytesIO(image["bytes"]))
        elif "np_array" in image:
            return Image.fromarray(image["np_array"])
        else:
            raise ValueError(f"Invalid image: {image}")
    else:
        raise ValueError(f"Invalid image: {image}")

def load_pretrained_model():
    # Load VITA preprocessing Tools
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_VITA_MODEL_NAME, use_fast=True)
    # tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_MODEL_NAME, use_fast=True)
    
    image_processor = CLIPImageProcessor.from_pretrained(DEFAULT_VIT_MODEL_NAME)
    
    audio_processor = load_audio_processor()
    
    return tokenizer, image_processor, audio_processor

from vl_load.vita.model.multimodal_encoder.whale.init_model import init_model
from transformers.utils.hub import get_file_from_repo
def load_audio_processor():
    with open(get_file_from_repo("VITA-MLLM/VITA-1.5_AudioEnc", "train.yaml"), "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    configs["cmvn_file"] = get_file_from_repo("VITA-MLLM/VITA-1.5_AudioEnc", "global_cmvn")

    audio_encoder = init_model(configs)
    audio_processor = audio_encoder.audio_processor
    return audio_processor

from gr00t.model.backbone.eagle2_hg_model.inference_eagle_repo import EagleProcessor

class VitaProcessor:
    def __init__(
        self,
    ):
        self.tokenizer, self.image_processor, self.audio_processor = load_pretrained_model()
        
        ## Execute once to adjust the tokenizer
        # num_added_tokens = self.tokenizer.add_tokens("<|ACT|>")
        # self.tokenizer.save_pretrained("/root/Isaac-GR00T/gr00t/tokenizer_with_act")
        
        self.eagle_processor = EagleProcessor()

    def prepare_input(self, message):
        ## process single data
        prompt=message["prompt"][1:][0] # currently only using single frame
        language = prompt["content"]
        images = prompt["image"]
        # print("here: ")
        # print(language)
        # print(len(images), images[0]["np_array"].shape)  # 2, (224, 224, 3)
        
        pixel_values = self.image_vita_fn(images)
        input_ids = self.text_vita_fn(language)
        # print(pixel_values.shape) # torch.Size([2, 3, 448, 448])
        
        
        # eagle data
        data_eagle = self.eagle_processor.prepare_input(message)
        
        # print("1 ", len(data_eagle["input_ids"][0]), len(input_ids))
        # print("2 ", data_eagle["pixel_values"][0].shape, pixel_values[0].shape)
        
        data = {
            "pixel_values": data_eagle["pixel_values"],
            "input_ids": data_eagle["input_ids"],
            "attention_mask": data_eagle["attention_mask"],
            
            "pixel_values_vita": pixel_values,
            "input_ids_vita": input_ids,
            # "attention_mask": attention_mask,
        }
        return data

    def collate_fn(self, all_examples):
        pixel_values_list = [ex["pixel_values_vita"] for ex in all_examples]
        input_ids_list = [ex["input_ids_vita"] for ex in all_examples]

        assert isinstance(pixel_values_list, List)
        assert isinstance(input_ids_list, List)

        pixel_values = torch.cat(pixel_values_list, dim=0)

        tokenized_batch = {
            "input_ids": [ip[0] for ip in input_ids_list],
        }

        padded = self.tokenizer.pad(tokenized_batch,
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                return_attention_mask=True,
                                return_tensors=None,
                                )
        input_ids = torch.tensor(padded["input_ids"], dtype=torch.long) # [batch, token]
        attention_mask = torch.tensor(padded["attention_mask"], dtype=torch.long)
        # print("here1: ", pixel_values.shape, input_ids.shape, attention_mask.shape)
        # torch.Size([16, 3, 448, 448]) torch.Size([8, 226]) torch.Size([8, 226])
        
        # eagle_data
        data_eagle = self.eagle_processor.collate_fn(all_examples)
        # print("here2: ", data_eagle["pixel_values"].shape, data_eagle["input_ids"].shape, data_eagle["attention_mask"].shape)
        # print("here2: ", pixel_values.shape)
        data = {
            "pixel_values": data_eagle["pixel_values"],
            "input_ids": data_eagle["input_ids"],
            "attention_mask": data_eagle["attention_mask"],
            
            "pixel_values_vita": pixel_values,
            "input_ids_vita": input_ids,
            "attention_mask_vita": attention_mask,
        }
        return BatchFeature(data)

    def image_vita_fn(self, sample):
        image=[]
        # print("sample: ", len(sample)) # two pic
        for s in sample:
            s = load_image(s)
            s=s.resize((448, 448))
            image_tensor = self.image_processor(images=s, return_tensors='pt')["pixel_values"]
            image.append(image_tensor)
        image = torch.cat(image, dim=0)

        return image

    def text_vita_fn(self, sample):
        question_prompt = "These two images are views of the same robotic arm from the front and its end effector position. Play the role of the robot arm in the picture. Based on the given task instructions, analyze the color and shape of the objects in front of you, and understand the relative position between the end effector of the robot arm and these objects. Provide as much information as possible to complete the task. Ignore objects that are not relevant to the task. Task instructions: "
        conv_mode="qwen2p5_instruct"
        input_ids_list=[]
        # print("sample: ", sample)
        # for question in sample:
        question = sample
        # print("question: ", question)
        qs = DEFAULT_IMAGE_TOKEN*2 + "\n" + question_prompt + question
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        # conv.append_message(conv.roles[1], "<|ACT|>")
        prompt = conv.get_prompt("image")
        # print("prompt: ", prompt)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX) # return list
        input_ids_list.append(input_ids)
        # print(input_ids_list)
        return input_ids_list
    
    def audio_mapping(self):
        # TODO: need to complete
        data_path = os.path.join(
            os.path.dirname(gr00t.__file__), "..", "dataset", "libero_spatial_no_noops_lerobot", "meta"
        )
        self.wave_path = os.path.join(
            os.path.dirname(gr00t.__file__), "..", "dataset", "wav_dataset", "libero_spatial"
        )

        self.tasks_dict = {}
        with open(data_path+'tasks.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.tasks_dict[data['task']] = data['task_index']
    
    def audio_vita_fn(self, sample):
        # TODO: need to complete
        question = sample
        # f"{self.tasks_dict[question]}.wav"