from typing import Literal

import lightning as L
from torch import Tensor
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from deeprec import losses

__all__ = ["BaseModel"]


class BaseModel(L.LightningModule):
    def __init__(
        self,
        weight_lat: bool = False,
        loss: Literal[
            "mae", "mse", "rmse", "huber", "nll_normal", "nll_laplace"
        ] = "huber",
        **kwargs,
    ):
        super().__init__()

        # Default parameters if not set during model init
        kwargs.setdefault("lr", 1e-3)
        kwargs.setdefault("weight_decay", 5e-3)
        kwargs.setdefault("eta_min", 1e-5)
        self.weight_lat = weight_lat
        self.save_hyperparameters()

        self.return_uncertainty = False

        match loss:
            case "mae":
                self.metric = losses.mae
            case "mse":
                self.metric = losses.mse
            case "rmse":
                self.metric = losses.rmse
            case "huber":
                self.metric = losses.huber
            case "nll_normal":
                self.metric = losses.nll_normal
                self.return_uncertainty = True
            case "nll_laplace":
                self.metric = losses.nll_laplace
                self.return_uncertainty = True
            case x:
                raise ValueError(f"Unknown loss '{x}'.")

        self.root_mean_squared_error = losses.rmse
        self.mean_absolute_error = losses.mae

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.eta_min,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        batch_idx: int,
    ):
        inputs: dict[str, Tensor] = batch["inputs"]
        target: Tensor = batch["target"]
        weight: Tensor = batch["weight"]

        preds: Tensor = self(**inputs)

        loss, rmse, mae = self._compute_metrics_and_loss(
            preds=preds,
            target=target,
            weight=weight,
        )

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_rmse", rmse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_mae", mae, on_step=False, on_epoch=True, sync_dist=True)

        # Log predicted uncertainty
        if self.return_uncertainty:
            uncertainty = preds[:, 1].mean()
            self.log(
                "train_uncertainty",
                uncertainty,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return loss

    def validation_step(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        batch_idx: int,
    ) -> None:
        self._shared_eval(batch, batch_idx, "val")

    def test_step(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        batch_idx: int,
    ) -> None:
        self._shared_eval(batch, batch_idx, "test")

    def predict_step(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        batch_idx: int,
    ) -> Tensor:
        inputs = batch["inputs"]
        return self(**inputs)

    def _shared_eval(
        self, batch: dict[str, Tensor | dict[str, Tensor]], batch_idx: int, prefix: str
    ) -> None:
        inputs: dict[str, Tensor] = batch["inputs"]
        target: Tensor = batch["target"]
        weight: Tensor = batch["weight"]

        preds: Tensor = self(**inputs)

        loss, rmse, mae = self._compute_metrics_and_loss(
            preds=preds,
            target=target,
            weight=weight,
        )

        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}_rmse", rmse, sync_dist=True)
        self.log(f"{prefix}_mae", mae, sync_dist=True)

        # Log predicted uncertainty
        if self.return_uncertainty:
            uncertainty = preds[:, 1].mean()
            self.log(f"{prefix}_uncertainty", uncertainty, sync_dist=True)

    def _compute_metrics_and_loss(
        self,
        preds: Tensor,
        target: Tensor,
        weight: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Compute loss conditionally based on whether weights are required
        if self.weight_lat:
            loss = self.metric(preds=preds, target=target, weight=weight)
        else:
            loss = self.metric(preds=preds, target=target)

        if self.return_uncertainty:
            # extract actual predictions (mu) before calculating default metrics
            preds = preds[:, 0]

        rmse = self.root_mean_squared_error(preds, target)
        mae = self.mean_absolute_error(preds, target)

        return loss, rmse, mae

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def scalar2matrix(x: Tensor, out_height: int) -> Tensor:
    """Convert scalar inputs of shape (N, C) into matrices of shape (N, C, H, H)"""
    return x.unflatten(1, (-1, 1, 1)).tile(1, 1, out_height, out_height)


def scalar2tensor(
    x: Tensor,
    out_depth: int,
    out_height: int,
) -> Tensor:
    """Convert scalar inputs of shape (N, C) into tensors of shape (N, C, L, H, H)"""
    return x.unflatten(1, (-1, 1, 1, 1)).tile(1, 1, out_depth, out_height, out_height)


def matrix2tensor(x: Tensor, out_depth: int) -> Tensor:
    """Convert matrix inputs of shape (N, C, H, H) into matrices of shape (N, C, L, H, H)"""
    return x.unflatten(1, (-1, 1)).tile(1, 1, out_depth, 1, 1)
