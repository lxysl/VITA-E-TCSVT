import torch
from torch import nn
from PIL import Image
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vl_load.vita.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    MAX_IMAGE_LENGTH,
)
from vl_load.vita.conversation import SeparatorStyle, conv_templates
from vl_load.vita.model.builder import load_pretrained_model
from vl_load.vita.util.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_audio_token,
    tokenizer_image_token,
)
from vl_load.vita.util.utils import disable_torch_init

class VITAModel(nn.Module):
    def __init__(
        self, 
        model_path="/root/vla/vl_load/checkpoints/VITA-1.5", 
        model_base=None, 
        image_path="/root/vla/vl_load/pic.jpg", 
        model_type="qwen2p5_instruct", 
        conv_mode="qwen2p5_instruct",
        question="Play the role of the robot arm in the picture. Based on the given task instructions, analyze the color and shape of the objects in front of you, and understand the relative position between the end effector of the robot arm and these objects. Provide as much information as possible to complete the task. Ignore objects that are not relevant to the task. Task instructions: close drawer.", 
        audio_path=None,
        frameCat=False):
        super().__init__()
        self.model_path = model_path
        self.model_base = model_base
        self.image_path = image_path
        self.model_type = model_type
        self.conv_mode = conv_mode
        self.qs = question
        self.frameCat = frameCat
        self.audio_path = audio_path
        if self.audio_path is not None:
            self.qs = "The first image is the egocentric camera view of the robot arm. The second image is the camera view at the end effector of this robot arm, which provides auxiliary information. Play as the robot arm in the first picture. Based on the given task instructions and the images, analyze the color and shape of the objects in front of you, and understand the relative position between the end effector of this robot arm and these objects. Provide as much information as possible to complete the task. Ignore objects that are not relevant to the task. Task instructions audio: "
        else:
            self.qs = "The first image is the egocentric camera view of the robot arm. The second image is the camera view at the end effector of this robot arm, which provides auxiliary information. Play as the robot arm in the first picture. Based on the given task instructions and the images, analyze the color and shape of the objects in front of you, and understand the relative position between the end effector of this robot arm and these objects. Provide as much information as possible to complete the task. Ignore objects that are not relevant to the task. Task instructions: close the drawer."
        self.image_path1 = "/root/Isaac-GR00T/top.png"
        self.image_path2 = "/root/Isaac-GR00T/wrist.png"
    
    def load_module(self):
        if self.frameCat:
            from vl_load.vita.util.data_utils_video_audio_neg_frameCat import dynamic_preprocess
        else:
            from vl_load.vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess

        # The number of visual tokens varies with the length of the video. "max_frames" is the maximum number of frames.
        # When the video is long, we will uniformly downsample the video to meet the frames when equal to the "max_frames".
        max_frames = MAX_IMAGE_LENGTH  # 100

        # Sampling Parameter
        temperature = 0.01
        top_p = None
        num_beams = 1

        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            self.model_path, self.model_base, model_name, self.model_type, device_map="cuda:0",
        )

        model.resize_token_embeddings(len(tokenizer))
        
        # import accelerate
        # from accelerate.hooks import add_hook_to_module, remove_hook_from_module

        # remove_hook_from_module(model, accelerate.hooks.AlignDevicesHook)
        # add_hook_to_module(self.model, accelerate.hooks.AlignDevicesHook(self.device_id))

        # # freeze model
        if True:
            for param in model.parameters():
                param.requires_grad = False


        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        image_processor = vision_tower.image_processor

        audio_encoder = model.get_audio_encoder()
        audio_encoder.to(dtype=torch.bfloat16)
        audio_processor = audio_encoder.audio_processor

        model.eval()

        if self.audio_path is not None:
            audio, audio_for_llm_lens = audio_processor.process(os.path.join(self.audio_path))
            audio_length = audio.shape[0]
            audio = torch.unsqueeze(audio, dim=0)
            audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
            audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
            audios = dict()
            audios["audios"] = audio.to(dtype=torch.bfloat16).cuda()
            audios["lengths"] = audio_length.to(dtype=torch.bfloat16).cuda()
            audios["lengths_for_llm"] = audio_for_llm_lens.cuda()
        else:
            audio = torch.zeros(400, 80)
            audio_length = audio.shape[0]
            audio_for_llm_lens = 60
            audio = torch.unsqueeze(audio, dim=0)
            audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
            audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
            audios = dict()
            audios["audios"] = audio.to(dtype=torch.bfloat16).cuda()
            audios["lengths"] = audio_length.to(dtype=torch.bfloat16).cuda()
            audios["lengths_for_llm"] = audio_for_llm_lens.cuda()

        if self.image_path is not None:
            image = Image.open(self.image_path1).convert("RGB")
            p_num=[2]
            image=image.resize((448, 448))
            image_tensor0 = model.process_images(image, model.config).to(
                dtype=model.dtype, device="cuda"
            )
            image1 = Image.open(self.image_path2).convert("RGB")
            image1=image1.resize((448, 448))
            image_tensor1 = model.process_images(image1, model.config).to(
                dtype=model.dtype, device="cuda"
            )
            image_tensor = torch.cat([image_tensor0, image_tensor1], dim=0)

            if self.audio_path:
                self.qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + self.qs + DEFAULT_AUDIO_TOKEN
            else:
                self.qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + self.qs
            modality = "image"
        else:
            image_tensor = torch.zeros((1, 3, 448, 448)).to(dtype=model.dtype, device="cuda")
            if self.audio_path:
                self.qs = self.qs + DEFAULT_AUDIO_TOKEN
            modality = "lang"

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], self.qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt(modality)

        input_ids_list=[]
        if self.audio_path:
            input_ids = (
                tokenizer_image_audio_token(prompt, tokenizer, IMAGE_TOKEN_INDEX)
            )
        else:
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX)
            )
        
        input_ids_list.append({"input_ids": input_ids})
        padded = tokenizer.pad(input_ids_list,
                                padding="longest",
                                max_length=tokenizer.model_max_length,
                                return_attention_mask=True,
                                return_tensors=None,
                                )
        input_ids = torch.tensor(padded["input_ids"], dtype=torch.long).cuda() # [batch, token]
        attention_mask = torch.tensor(padded["attention_mask"], dtype=torch.long).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                audios=audios,
                do_sample=False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                shared_v_pid_stride=None#2#16#8#4#1#None,
            )
        output_ids = output_ids.sequences
        input_token_len = input_ids.shape[1]
        if self.model_type == "mixtral-8x7b":
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
                output_ids = output_ids[:, input_token_len:]
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]

        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        print(outputs)


if __name__ == "__main__":
    model = VITAModel()
    model.load_module()