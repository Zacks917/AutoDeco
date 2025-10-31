# AutoDeco

We utilize the NVIDIA NeMo framework for training large language models:

- GPT-OSS-120B
- Qwen3-235B-A22B
- DeepSeek-V3.1-Terminus

Due to the complexity of setting up the NeMo/Megatron environment, we have created a new submodule dedicated to explaining our training methodology.

## 1. Environment Setup

We use [uv](https://github.com/astral-sh/uv) to manage the Python environment. It is recommended to first install the base dependencies by following the steps below:

```bash
mkdir -p /root/py_envs/ && cd /root/py_envs/ 
uv init base && cd base && uv venv --python 3.12.7 
source .venv/bin/activate 
uv pip install setuptools_rust wheel --upgrade 
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 
uv pip install transformers peft accelerate safetensors trl liger-kernel 
uv pip install ray[all] deepspeed deepspeed-kernels &&  \
uv pip install xformers --index-url https://download.pytorch.org/whl/cu128 --no-build-isolation 
uv pip install flash-attn --no-build-isolation 
uv pip install bitsandbytes sentencepiece torchmetrics pandas openpyxl 
uv pip install wandb tensorboard 
uv pip install hf_transfer tyro orjson dotenv toml scipy openai
uv pip install numpy==1.26.4 
```

Since GPT-OSS-120B has high version requirements for NeMo, we will install `NeMo` and `Megatron-Core` from source. 
The relevant files have been prepared for you in the `autodeco_megatron/py_packages` directory.

1. Install `apex` and `transformer-engine`
2. 
```bash
cd /root/py_envs/base 
unzip apex-25.09.zip

APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install -v --no-build-isolation .
uv pip install --no-build-isolation transformer_engine[pytorch]
```

2. Install `megatron-core` and `NeMo`

```bash
cd /root/py_envs/base 
unzip Megatron-LM-main.zip
unzip NeMo-main.zip

cd Megatron-LM-main
uv pip install .

cd NeMo-main
uv pip install ".[all]"
```

## 2. Model Download and Format Conversion

You can download the required models from HuggingFace. Since the model files are large, to avoid unnecessary loading time, it is recommended that you pre-download them to the local disk of the training server.

Please note that for DeepSeek-V3.1-Terminus, the `transformers` currently does not support directly loading model files in FP8 format. You need to first perform a `FP8 -> BF16` format conversion.

```bash 
cd nemo_converter/deepseek_v3
# https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.py
python3 fp8_cast_bf16.py --input <> --output <>
```

As shown below, in the `nemo_converter` folder, we have prepared the format conversion files for you:

```textmate
.
├── deepseek_v3
│         ├── fp8_cast_bf16.py
│         ├── kernel.py
│         └── nemo_import_deepseek_v3.py
├── gpt_oss
│         ├── nemo_import_gpt_oss_120b.py
│         └── nemo_import_gpt_oss_20b.py
├── llama_31
│         └── nemo_import_llama31_nemotron_8b_v1.py
├── qwen25
│         └── nemo_import_qwen25_7b.py
├── qwen3
│     ├── nemo_import_qwen3_235_a22b.py
│     └── nemo_import_qwen3_30_a3b.py
└── nemo_export.py
```

You can use the corresponding `nemo_import_<model>.py` script to convert the model from HuggingFace format to NeMo format, for example:

```bash
python3 nemo_converter/deepseek_v3/nemo_import_deepseek_v3.py --input <hf model path> --output <nemo format save path>
```

## 3. 训练

We recommend using at least 8 machines (with a total of 64 GPUs) to train the above model. For the training configuration of each model, please refer to the `end2end.sh` file in the scripts folder.

The specific steps are as follows:


```textmate
.
├── deepseek
│         └── end2end.sh
├── gpt_oss
│         └── end2end_120b.sh
└── qwen3
    └── end2end_235b_a22b.sh
```

1. First, configure the following variables in the corresponding `end2end.sh`:

```textmate
NEMO_MODEL_PATH=""
HF_MODEL_PATH=""

TRAIN_FP=""

RUN_NAME=""
SAVE_DIR=""

MASTER_ADDR=""
```

2. Start the corresponding `end2end.sh` training script on each machine. Please ensure that `MASTER_ADDR` and `NODE_RANK` are configured correctly.

3. After training is completed, the weight files corresponding to the temperature head and top-p head will be saved under
   `SAVE_DIR/RUN_NAME/<TRAIN TASK START TIMESTAMP>/checkpoints/<CHECKPOINT>`.
   You can use the `nemo_export.py` script to convert the model weights from NeMo format back to HuggingFace format and extract the head weights, for example:

```bash
python3 nemo_converter/nemo_export.py --input <nemo checkpoint dir> --output <hf save dir> --target hf-peft
```

4. Finally, by running `AutoDeco/script/merge_autodeco.py`, the head weights can be merged into the main model.
