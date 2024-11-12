"""Containes losses for PyTorch model training and evaluation."""

import torch
from torch import Tensor
from torch.nn.functional import huber_loss, l1_loss, mse_loss


def mae(preds: Tensor, target: Tensor, weight: Tensor | None = None) -> Tensor:
    """Latitude weighted mean absolute error."""
    if weight is not None:
        loss = l1_loss(preds, target, reduction="none")
        loss = (loss * weight / weight.mean()).mean()
    else:
        loss = l1_loss(preds, target)
    return loss


def mse(preds: Tensor, target: Tensor, weight: Tensor | None = None) -> Tensor:
    """Latitude weighted mean squared error."""
    if weight is not None:
        loss = mse_loss(preds, target, reduction="none")
        loss = (loss * weight / weight.mean()).mean()
    else:
        loss = mse_loss(preds, target)
    return loss


def rmse(preds: Tensor, target: Tensor, weight: Tensor | None = None) -> Tensor:
    """Latitude weighted root mean squared error."""
    if weight is not None:
        loss = torch.sqrt(mse_loss(preds, target, reduction="none"))
        loss = (loss * weight / weight.mean()).mean()
    else:
        loss = torch.sqrt(mse_loss(preds, target))
    return loss


def nse(
    preds: Tensor, target: Tensor, std: Tensor, weight: Tensor | None = None
) -> Tensor:
    """Modified Nash-Sutcliffe model efficiency score.
    Contrary to the classical definition of NSE, less is better:
    NSE' = 1 - NSE"""
    EPS = 0.1
    squared_error = mse_loss(preds, target, reduction="none")
    scaled_loss = squared_error / (std + EPS) ** 2
    if weight is not None:
        scaled_loss = scaled_loss * weight / weight.mean()
    return scaled_loss.mean()


def huber(preds: Tensor, target: Tensor, weight: Tensor | None = None) -> Tensor:
    """Huber loss"""
    if weight is not None:
        loss = huber_loss(preds, target, reduction="none")
        loss = (loss * weight / weight.mean()).mean()
    else:
        loss = huber_loss(preds, target)
    return loss


def nll_normal(preds: Tensor, target: Tensor, weight: Tensor | None = None) -> Tensor:
    """Negative loss-likelihood for a normal distribution, for uncertainty prediction according to

    Lakshminarayana, Pritzel, Blundell,
    Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles,
    https://arxiv.org/abs/1612.01474, arXiv, 2017

    The constant log term is omitted.
    """

    EPS = 1e-8

    # Split preds into values (mu) and variances (var)
    mu, var = preds.T
    var = var + EPS

    squared_error = mse_loss(mu, target, reduction="none")
    loss = 0.5 * var.log() + 0.5 * (squared_error / var)

    if weight is not None:
        loss = loss * weight / weight.mean()
    return loss.mean()


def nll_laplace(preds: Tensor, target: Tensor, weight: Tensor | None = None) -> Tensor:
    """Negative loss-likelihood for a Laplace distribution.
    The constant log term is omitted.
    """

    EPS = 1e-8

    # Split preds into values (mu) and scale parameters (b)
    mu, b = preds.T
    b = b + EPS

    absolute_error = l1_loss(mu, target, reduction="none")
    loss = b.log() + (absolute_error / b)

    if weight is not None:
        loss = loss * weight / weight.mean()
    return loss.mean()
