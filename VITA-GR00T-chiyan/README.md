# README-vita_gr00t

## Installation Guide
### gr00t
Clone the repo:

```sh
git clone
cd Isaac-GR00T
```

Create a new conda environment and install the dependencies. We recommend Python 3.10:

> Note that, please make sure your CUDA version is 12.4. Otherwise, you may have a hard time with properly configuring flash-attn module.

```sh
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools
pip install -e .
pip install --no-build-isolation flash-attn==2.7.1.post4 
```

### vita
```sh
cd gr00t/model/vl_load
pip install -r requirements.txt
```

**note:** Please refer to the gr00t and vita projects to download checkpoints to `./nvidia/` and `./gr00t/model/vl_load/checkpoints` respectively.

### Solutions to egl related problems
Refer to the content of this link: https://iwiki.woa.com/p/4014435000

## 训练
``` bash
python scripts/libero_inference_vita.py: vita_gr00t单数据集微调
python scripts/gr00t_finetune_vita_mixed.py: vita_gr00t多数据集微调
python scripts/gr00t_finetune.py: gr00t模型微调
```

## mse画图
``` bash
python scripts/eval_policy.py
```

## 数据集目录
参考`/root/Isaac-GR00T/dataset`

## 评测
### 评测服务端
想要在libero中评测需要搭配客户端使用。
``` bash
python scripts/inference_service.py: 原版调试用
python scripts/inference_service_RTC.py: 对vita_gr00t模型评测时可启动RTC推理用，内含服务端推理逻辑
python scripts/libero_inference.py: gr00t模型评测用
python scripts/libero_inference_vita.py: vita_gr00t模型评测用
```

### openpi-libero仿真客户端配置
1. 拉取openpi https://github.com/Physical-Intelligence/openpi
2. 参考其中openpi/examples/libero中的README配置环境。
3. 拷贝gr00t仿真交互文件至openpi中：
``` bash
cp Isaac-GR00T/libero_simulation/main* ~/openpi/examples/libero/
cp Isaac-GR00T/libero_simulation/zmq_gr00t.py ~/openpi/packages/openpi-client/src/openpi_client/
```
4. 启动仿真交互客户端：
``` bash
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
python examples/libero/main.py
```