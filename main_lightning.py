import argparse
import importlib
import os
import random
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
os.environ["DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import lightning as L
import streaming
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy
from omegaconf import OmegaConf
from torch import distributed as dist
from torch.utils.data import DataLoader

from data.collate_fn import collate_fn_train
from data.get_datasets import get_datasets
from models.get_models import get_model_dict
from models.lightning.callbacks.result_callback import ResultSaveCallback
from models.lightning.callbacks.gc_callback import ScheduledGarbageCollector


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = True


def _create_folders(save_folder):
    image_folder = os.path.join(save_folder, "valid_images")
    os.makedirs(image_folder, exist_ok=True)

    umap_folder = os.path.join(save_folder, "umap")
    os.makedirs(umap_folder, exist_ok=True)

    model_folder = os.path.join(save_folder, "models")
    os.makedirs(model_folder, exist_ok=True)

    metrics_folder = os.path.join(save_folder, "metrics")
    os.makedirs(metrics_folder, exist_ok=True)
    return model_folder


def app(config):
    exp_folder = os.path.join(
        config["exp_folder"], config["exp_name"], config["exp_mode"]
    )
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder, exist_ok=True)

    model_name = "_".join(
        [
            k + "_" + str(v["name"])
            for k, v in config["model"].items()
            if k in ["encoder", "decoder", "ssl_model", "ae_model"]
        ]
    )
    exp_folder_config = os.path.join(exp_folder, model_name)

    if not os.path.exists(exp_folder_config):
        os.makedirs(exp_folder_config, exist_ok=True)
    with open(os.path.join(exp_folder_config, "config_exp.yaml"), "w") as fp:
        yaml.dump(config, fp)

    model_folder = _create_folders(exp_folder_config)

    wandb_logger = (
        WandbLogger(
            project=config["exp_name"],
            name=config["exp_mode"],
            config=config,
            dir=exp_folder_config,
        )
        if config["log_wandb"]
        else None
    )

    n_gpus_per_node = torch.cuda.device_count()
    num_gpus = int(os.environ["WORLD_SIZE"])
    num_nodes = int(num_gpus // n_gpus_per_node)

    print(
        f"num_nodes: {num_nodes}, num_gpus: {num_gpus}, n_gpus_per_node: {n_gpus_per_node}",
        flush=True,
    )

    # # Fingers crossed PTL has a guard around init_process_group...
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group(backend="nccl")

    if "16" in config["trainer"]["precision"]:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        print(f"FlashAttention enabled ...")
    elif "32" in config["trainer"]["precision"]:
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        print("Memory-efficient Attention enabled ...")

    train_batch_size = config["train"]["train_batch_size"]
    val_batch_size = config["train"]["test_batch_size"]
    image_per_sample = config["data"]["args"]["n_cells"]
    train_device_microbatch_size = int(train_batch_size // num_gpus) * image_per_sample
    val_device_microbatch_size = int(val_batch_size // num_gpus) * image_per_sample

    datasets, transforms = get_datasets(
        config["data"], train_device_microbatch_size, val_device_microbatch_size
    )
    train_dataloader = DataLoader(
        datasets[0],
        batch_size=train_batch_size // num_gpus,
        collate_fn=collate_fn_train,
        num_workers=config["num_workers"],
        persistent_workers=True,
        pin_memory=config["pin_memory"],
    )
    valid_loader = DataLoader(
        datasets[1],
        batch_size=val_batch_size // num_gpus,
        collate_fn=collate_fn_train,
        num_workers=config["num_workers"],
        persistent_workers=True,
        pin_memory=config["pin_memory"],
    )

    model_dict = get_model_dict(config["model"])
    loss_weights = torch.ones(len(datasets[0].unique_cats))
    model_dict.update(
        {
            "color_channels": datasets[0].color_channels,
            "save_folder": exp_folder_config,
            "num_classes": datasets[0].num_classes,
            "categories": datasets[0].unique_cats,
            "class_weights": loss_weights,
            "batches_per_epoch": len(train_dataloader),
            "transforms": transforms[0],
            "transforms2": (
                transforms[1] if config["train"]["pl_module"] != "BaseMAE" else None
            ),
            "valid_transforms": transforms[2],
        }
    )
    ## update learning rate
    model_dict["init_lr"] = (
        model_dict["init_lr"] * train_device_microbatch_size * num_gpus
    ) / 256

    pl_model_module = importlib.import_module("models.lightning")
    model = getattr(pl_model_module, config["train"]["pl_module"])(**model_dict)

    if config["trainer"]["strategy"] == "ddp":
        strategy = DDPStrategy(process_group_backend="nccl")
    elif config["trainer"]["strategy"] == "deepspeed":
        strategy = DeepSpeedStrategy(stage=2)

    all_callbacks = []
    result_callback = ResultSaveCallback(plot_metrics=True, plot_feats=False)
    all_callbacks.append(result_callback)

    if config["trainer"]["gc_interval"] > 0:
        gc_callback = ScheduledGarbageCollector(
            gen_1_batch_interval=config["trainer"]["gc_interval"]
        )
        all_callbacks.append(gc_callback)

    model_checkpoint_ap = ModelCheckpoint(
        dirpath=model_folder,
        filename="best_model_ap",
        monitor="val_metrics/total_ml_auprc",
        verbose=True,
        save_last=True,
        save_top_k=1,
        mode="max",
        enable_version_counter=False,
    )
    all_callbacks.append(model_checkpoint_ap)
    model_checkpoint_mlrap = ModelCheckpoint(
        dirpath=model_folder,
        filename="best_model_mlrap",
        monitor="val_metrics/total_mlrap",
        verbose=True,
        save_last=True,
        save_top_k=1,
        mode="max",
        enable_version_counter=False,
    )
    all_callbacks.append(model_checkpoint_mlrap)

    trainer = L.Trainer(
        default_root_dir=exp_folder_config,
        accelerator="gpu",
        num_nodes=num_nodes,
        strategy=strategy,
        devices="auto",
        check_val_every_n_epoch=config["trainer"]["valid_every"],
        max_epochs=model.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=50,
        sync_batchnorm=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=all_callbacks,
        num_sanity_val_steps=0,
        precision=config["trainer"]["precision"],
    )

    ckpt_path = f"{model_folder}/{config['train']['ckpt_path']}"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
    print(f"Checkpoint path: {ckpt_path}", flush=True)

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_loader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="SearchFirst config file path")
    argparser.add_argument("-c", "--config", help="path to configuration file")
    argparser.add_argument(
        "-r", "--random_seed", help="random_seed", default=42, type=int
    )

    args = argparser.parse_args()

    config_path = args.config
    random_seed = args.random_seed

    set_random_seed(random_seed)

    om_config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(om_config)

    # streaming.base.util.clean_stale_shared_memory()
    print(config)
    print(config["exp_name"])
    print(config["exp_mode"])
    app(config)
