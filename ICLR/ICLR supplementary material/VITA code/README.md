# VITA-E: A Dual-Model Framework for Real-Time, Interruptible, and Concurrent Human-Robot Interaction

## 👀 VITA-E Overview

### 🌟 Spotlight in VITA-E

We are excited to present **VITA-E**, which incorporates a series of advancements:

1. **Dual-Model Framework for Seamless Interaction**. VITA-E introduces a groundbreaking dual-model core, where an "Active Model" executes tasks while a "Listening Model" stands ready for new commands.

2. **Innovative "Model-as-Controller" Paradigm**. We pioneer a "model-as-controller" approach where the Vision-Language Model is fine-tuned to generate special tokens that function as direct system-level commands, enabling precise, reliable, and immediate control over system actions.

3. **Smooth Human-Computer Interaction**. By this manner, VITA-E supports smooth two-way voice interaction, allows replies while executing, voice interruption during actions, and natural action transition. Besides, VITA-E supports both English and Chinese.

4. **Strong Performance in Critical Interactive Scenarios**. Tested on a physical humanoid robot, VITA-E demonstrated exceptional reliability and responsiveness. It achieves a high success rate across multiple interactive and operational tasks and is compatible with a wide range of mainstream VLA models.


## 📈 Experimental Results

- **Success rate comparison of VITA-E and baseline models on two fundamental manipulation tasks.**

<p align="center">
    <img src="./asset/vita-e-results.png" width="100%" height="100%">
</p>


## 📐 Inference

Install conda environment.

```
git clone https://github.com/VITA-MLLM/VITA-E
cd VITA-E
conda create -n vitae python=3.10 -y
conda activate vitae
pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

Download the required model weights to local path: (1) VITA-E VLM; (2) VITA-E with action expert.

```bash
huggingface-cli download VITA-VLA/vita_vla_finetune --local-dir checkpoints/vita_vla_finetune
huggingface-cli download VITA-VLA/vita_gr00t_robot_head --local-dir checkpoints/vita_gr00t_robot_head
```

Run the inference script.

```bash
python inference_vita_e.py --model_path_vlm checkpoints/vita_vla_finetune --model_path_policy checkpoints/vita_gr00t_robot_head
```

### 📍 Demo

#### Web Demo

You can interact with our VITA-E web demo with mocked robot state data to experience the features, with no need of any embodied robot entity. (A total of 48 GB GPU memory is needed.)

Prepare a VAD (Voice Activity Detection) module. 
You can choose to download [silero_vad.onnx](https://github.com/snakers4/silero-vad/tree/v4.0/files) and [silero_vad.jit](https://github.com/snakers4/silero-vad/tree/v4.0/files), and place these files in the `./demo/wakeup_and_vad/resource/` directory.

```bash
python -m demo.server_vla_vita --model_path_vlm checkpoints/vita_vla_finetune --model_path_policy checkpoints/vita_gr00t_robot_head --ip 0.0.0.0 --port 8081
```

Wait about three minutes to completely load all modules. Open `127.0.0.1:8081` website on you server and enjoy it.

#### Real Robot Demo

Deploy server script on your server.

```bash
python -m demo.server_vla_vita --model_path_vlm checkpoints/vita_vla_finetune --model_path_policy checkpoints/vita_gr00t_robot_head --ip 0.0.0.0 --port 8081
```

Start client script on the robot client.

```bash
cd demo
python vla_robot_client.py
```

## ⭐ Training VLM

The training pipeline of VLM is same to VITA-1.5.

#### Requirements and Installation
```
conda create -n vita python=3.10 -y
conda activate vita
pip install --upgrade pip
pip install -r train_requirements.txt
pip install flash-attn --no-build-isolation
```

#### Data Preparation
- An example json file of the training data:
```
[
    ...
    {
        "set": "sharegpt4",
        "id": "000000000164",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n<audio>\n"
            },
            {
                "from": "gpt",  // follow the setting of llave, "gpt" is only used to indicate that this is the ground truth of the model output
                "value": "This is a well-organized kitchen with a clean, modern aesthetic. The kitchen features a white countertop against a white wall, creating a bright and airy atmosphere. "
            }
        ],
        "image": "coco/images/train2017/000000000164.jpg",
        "audio": [
            "new_value_dict_0717/output_wavs/f61cf238b7872b4903e1fc15dcb5a50c.wav"
        ]
    },
    ...
]
```

- The `set` field is used to retrieve the image or video folder for data loading. You should add its key-value pair to the `FolderDict` in [./vita/config/dataset_config.py](./vita/config/dataset_config.py):
```
AudioFolder = ""
FolderDict = {
    #### NaturalCap
    "sharegpt4": "",
}
#### NaturalCap
ShareGPT4V = {"chat_path": ""}
```

- Set the JSON path for `"chat_path"` in the corresponding dictionary in [./vita/config/dataset_config.py](./vita/config/dataset_config.py).
- Set the audio folder path for `AudioFolder` in [./vita/config/dataset_config.py](./vita/config/dataset_config.py).
- Add the data class in `DataConfig` in [./vita/config/init.py](./vita/config/__init__.py):
```
from .dataset_config import *

NaturalCap = [ShareGPT4V]

DataConfig = {
    "Pretrain_video": NaturalCap,
}
```

#### Continual Training
- Download the required weights: (1) [VITA-1.5 checkpoint](https://huggingface.co/VITA-MLLM/VITA-1.5/tree/main), (2) [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px), and (3) [Our pretrained audio encoder](https://huggingface.co/VITA-MLLM/VITA-1.5/tree/main/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning) in Stage-2 audio-language alignment (refer to Fig. 3 in the paper).

- Replace the paths in [./script/train/finetuneTaskNeg_qwen_nodes.sh](https://github.com/BradyFU/VITA-Temp/blob/main/script/train/finetuneTaskNeg_qwen_nodes.sh):
```
    ...
    --model_name_or_path VITA1.5_ckpt \
    ...
    --vision_tower InternViT-300M-448px \
    ...
    --audio_encoder audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning \
    ...
```

- Execute the following commands to start the training process:

```
export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTPUT_DIR=/mnt/cfs/lhj/videomllm_ckpt/outputs/vita_video_audio
bash script/train/finetuneTaskNeg_qwen_nodes.sh ${OUTPUT_DIR}
```

## ⭐ Training Action Expert

We follow Isaac-GR00T architecture and use its pre-trained diffusion action expert to build our VITA-E model.

#### Requirements and Installation
```
conda create -n gr00t python=3.10
conda activate gr00t
cd Isaac-GR00T
pip install --upgrade setuptools
pip install -e .
pip install --no-build-isolation flash-attn==2.7.1.post4 
```

#### Finetune

* set the VITA model path: `Isaac-GR00T/gr00t/model/gr00t_vita.py: Line 86`
* set the VLA dataset path: `Isaac-GR00T/scripts/gr00t_finetune_vita_real_robot.py: Line 39`
* set the output path: `Isaac-GR00T/scripts/gr00t_finetune_vita_real_robot.py: Line 42`

```
python scripts/gr00t_finetune_vita_real_robot.py
```

## &#x1F4E3; Statement

**VITA is trained on large-scale open-source corpus, and its output has randomness. Any content generated by VITA does not represent the views of the model developers. We are not responsible for any problems arising from the use, misuse, and dissemination of VITA, including but not limited to public opinion risks and data security issues.**

