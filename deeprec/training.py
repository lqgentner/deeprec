"""Utilities to support the training of PyTorch models"""

from pathlib import Path

import lightning as L
import torch
import torchinfo
from omegaconf import OmegaConf

import wandb


def define_wandb_metrics() -> None:
    """Defines Weights & Biases custom metrics"""
    wandb.define_metric("train_loss", step_metric="trainer/global_step", summary="min")
    wandb.define_metric("val_loss", step_metric="trainer/global_step", summary="min")
    wandb.define_metric("test_loss", step_metric="trainer/global_step", summary="min")
    wandb.define_metric("train_rmse", step_metric="trainer/global_step", summary="min")
    wandb.define_metric("val_rmse", step_metric="trainer/global_step", summary="min")
    wandb.define_metric("test_rmse", step_metric="trainer/global_step", summary="min")
    wandb.define_metric("train_mae", step_metric="trainer/global_step", summary="min")
    wandb.define_metric("val_mae", step_metric="trainer/global_step", summary="min")
    wandb.define_metric("test_mae", step_metric="trainer/global_step", summary="min")


def load_config(file: str | Path) -> dict:
    """Loads a YAML configuration file with OmegaConf"""
    # Register resolver to parse length of objects (${len:${x}}) in YAML
    OmegaConf.register_new_resolver("len", len, replace=True)
    # Load from YAML and convert DictConfig to native dict
    return OmegaConf.to_object(OmegaConf.load(file))


def model_summary(
    model: L.LightningModule, data_config: dict, **kwargs
) -> torchinfo.ModelStatistics:
    """Returns Keras-like model summary"""
    input_data = {}
    if "scalar_input_vars" in data_config:
        scalar_size = (
            data_config["batch_size"],
            len(data_config["scalar_input_vars"]),
        )
        input_data["scalar_inputs"] = torch.zeros(scalar_size)
    if "vector_input_vars" in data_config:
        vector_size = (
            data_config["batch_size"],
            len(data_config["vector_input_vars"]),
            data_config["time_window"],
        )
        input_data["vector_inputs"] = torch.zeros(vector_size)
    if "matrix_input_vars" in data_config:
        matrix_size = (
            data_config["batch_size"],
            len(data_config["matrix_input_vars"]),
            data_config["space_window"],
            data_config["space_window"],
        )
        input_data["matrix_inputs"] = torch.zeros(matrix_size)
    if "tensor_input_vars" in data_config:
        # Add tensors to input_data
        tensor_size = (
            data_config["batch_size"],
            len(data_config["tensor_input_vars"]),
            data_config["time_window"],
            data_config["space_window"],
            data_config["space_window"],
        )
        input_data["tensor_inputs"] = torch.zeros(tensor_size)

    return torchinfo.summary(model, input_data=input_data, **kwargs)
