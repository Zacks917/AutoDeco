import json
import os.path
import sys

import tyro

sys.path.append(".")
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Literal

import torch

from lightning.pytorch.loggers import WandbLogger
from nemo import lightning as nl

from nemo.collections import llm

from megatron.core.dist_checkpointing.validation import StrictHandling

from nemo.lightning.pytorch.plugins.mixed_precision import MegatronMixedPrecision
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule

from nemo.utils.exp_manager import TimingCallback
from megatron.core.optimizer import OptimizerConfig

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig

TokenizerType = Any

from src.data.auto_deco_dataset import AutoDecoDataModule
from src.pl_models.llama import AutoDecoLlamaModelNeMo, AutoDecoLLama31Config8B
from src.pl_models.qwen25 import AutoDecoQwen2ModelNeMo, AutoDecoQwen25Config7B
from src.pl_models.qwen3 import AutoDecoQwen3ModelNeMo, AutoDecoQwen3Config30BA3B, AutoDecoQwen3Config235BA22B
from src.pl_models.gpt_oss import AutoDecoGPTOSSModelNeMo, AutoDecoGPTOSSConfig20B, AutoDecoGPTOSSConfig120B
from src.pl_models.deepseek_v3 import AutoDecoDeepSeekV3ModelNeMo, AutoDecoDeepSeekV3Config
from src.pl_models.common import FreezeMainLLMParameters


@dataclass
class Args:
    nnodes: int
    nproc_per_node: int

    model_type: Literal[
        "llama-31-8b",
        "qwen25-7b",
        "qwen3-30b-a3b",
        "qwen3-235b-a22b",
        "gpt-oss-20b",
        "gpt-oss-120b",
        "deepseek-v3"
    ]
    nemo_model_path: str
    hf_model_path: str
    model_head_path: str
    train_temp: bool
    train_top_p: bool
    learning_rate: float
    weight_decay: float

    tensor_model_parallel_size: int
    expert_model_parallel_size: int
    context_parallel_size: int
    pipeline_model_parallel_size: int

    train_fp: str
    eval_fp: str
    max_length: int
    save_dir: str
    micro_batch_size: int
    global_batch_size: int
    max_steps: int
    max_epochs: int
    num_train_items: int

    warmup_steps: int

    run_name: str

    num_layers_in_first_pipeline_stage: Optional[int] = field(default=None)  # 6
    num_layers_in_last_pipeline_stage: Optional[int] = field(default=None)  # 1


if __name__ == '__main__':
    args: Args = tyro.cli(Args)

    if args.max_steps == -1:
        args.max_steps = (args.num_train_items // args.global_batch_size + 1) * args.max_epochs

    if args.num_layers_in_first_pipeline_stage == -1 or args.pipeline_model_parallel_size == 1:
        args.num_layers_in_first_pipeline_stage = None
    if args.num_layers_in_last_pipeline_stage == -1 or args.pipeline_model_parallel_size == 1:
        args.num_layers_in_last_pipeline_stage = None

    os.system(f"mkdir -p {os.path.join(args.save_dir, args.run_name)}")
    with open(os.path.join(args.save_dir, args.run_name, "train_settings.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(asdict(args), ensure_ascii=False, indent=2))

    datamodule = AutoDecoDataModule(
        hf_model_path=args.hf_model_path,
        fp=args.train_fp,
        max_length=args.max_length,
        num_workers=8,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
    )

    config_kwargs = dict(
        init_model_with_meta_device=False,
        calculate_per_token_loss=True,
        train_temp_head=args.train_temp,
        train_top_p_head=args.train_top_p,
        output_hidden_states=False,
        model_head_path="",
        recompute_granularity="selective",
    )

    if args.model_type == "llama-31-8b":
        model = AutoDecoLlamaModelNeMo(config=AutoDecoLLama31Config8B(**config_kwargs))
    elif args.model_type == "qwen25-7b":
        model = AutoDecoQwen2ModelNeMo(config=AutoDecoQwen25Config7B(**config_kwargs))
    elif args.model_type == "qwen3-30b-a3b":
        model = AutoDecoQwen3ModelNeMo(config=AutoDecoQwen3Config30BA3B(**config_kwargs))
    elif args.model_type == "qwen3-235b-a22b":
        config_kwargs.update({
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1
        })
        model = AutoDecoQwen3ModelNeMo(config=AutoDecoQwen3Config235BA22B(**config_kwargs))
    elif args.model_type == "gpt-oss-20b":
        model = AutoDecoGPTOSSModelNeMo(config=AutoDecoGPTOSSConfig20B(**config_kwargs))
    elif args.model_type == "gpt-oss-120b":
        config_kwargs.update({
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1
            # "recompute_modules": ["mla_up_proj", "layernorm"]
        })
        model = AutoDecoGPTOSSModelNeMo(config=AutoDecoGPTOSSConfig120B(**config_kwargs))
    elif args.model_type == "deepseek-v3":
        config_kwargs.update({
            "recompute_granularity": "selective",
            "recompute_modules": ["mla_up_proj", "layernorm"]
        })
        model = AutoDecoDeepSeekV3ModelNeMo(config=AutoDecoDeepSeekV3Config(**config_kwargs))
    else:
        assert False, args.model_type

    resume = nl.AutoResume(restore_config=nl.RestoreConfig(path=args.nemo_model_path))

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=args.context_parallel_size,
        num_layers_in_first_pipeline_stage=args.num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=args.num_layers_in_last_pipeline_stage,
        sequence_parallel=args.tensor_model_parallel_size > 1 and args.expert_model_parallel_size > 1,
        gradient_as_bucket_view=True,
        ckpt_load_strictness=StrictHandling.IGNORE_ALL,
        ckpt_parallel_load=True,
        ckpt_async_save=True,
        ddp=DistributedDataParallelConfig(
            overlap_param_gather=False,
            overlap_grad_reduce=False,
        ),
    )

    scheduler = CosineAnnealingScheduler(
        warmup_steps=args.warmup_steps,
        constant_steps=50,
        max_steps=args.max_steps,
        min_lr=2e-8,
    )

    extra_optimizer_config = {}
    if args.model_type in {"deepseek-v3", "qwen3-235-a22b"}:
        extra_optimizer_config = dict(
            use_precision_aware_optimizer=True,
            main_params_dtype=torch.float32,
            main_grads_dtype=torch.bfloat16,
            exp_avg_dtype=torch.bfloat16,
            exp_avg_sq_dtype=torch.bfloat16
        )

    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        bf16=True,
        fp16=False,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        **extra_optimizer_config
    )

    optimizer = MegatronOptimizerModule(config=optimizer_config, lr_scheduler=scheduler)

    model_checkpoint = nl.ModelCheckpoint(
        monitor=None,
        save_last=False,
        save_top_k=-1,
        every_n_epochs=1,
        filename="{model_name}-{step}-{consumed_samples}",
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
        save_weights_only=True,
        always_save_context=True,
        save_context_on_train_end=True
    )

    logger = nl.NeMoLogger(
        name=args.run_name,
        ckpt=model_checkpoint,
        wandb=WandbLogger(name=args.run_name, save_dir=args.save_dir),
        # tensorboard=TensorBoardLogger(name=args.run_name, save_dir=args.save_dir),
        log_global_rank_0_only=True,
        # log_local_rank_0_only=True,
        log_dir=args.save_dir,
        update_logger_directory=False,
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        num_nodes=args.nnodes,
        devices=args.nproc_per_node,

        limit_test_batches=None,
        limit_val_batches=None,
        log_every_n_steps=1,
        # max_steps 不能设为 -1 ，会影响 lr_scheduler
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        plugins=MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=True,
            grad_reduce_in_fp32=True
        ),
        strategy=strategy,
        val_check_interval=0,
        callbacks=[TimingCallback()],
        check_val_every_n_epoch=10,
        default_root_dir=args.save_dir,
        logger=False
    )

    freeze_transform = FreezeMainLLMParameters(train_temp_head=args.train_temp, train_top_p_head=args.train_top_p)

    output = llm.finetune(
        model=model,
        data=datamodule,
        trainer=trainer,
        log=logger,
        optim=optimizer,
        resume=resume,
        tokenizer="model",
        peft=freeze_transform
    )
