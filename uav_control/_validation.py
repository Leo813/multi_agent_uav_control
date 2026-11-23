"""Small shared checks for the eager, float32/float64 mathematical APIs."""

import math
from numbers import Real

import torch


def finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")


def floating(tensor: torch.Tensor, name: str, ndim: int | None = None) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64")
    if tensor.numel() == 0 or (ndim is not None and tensor.ndim != ndim):
        raise ValueError(f"{name} has an invalid or empty shape: {tuple(tensor.shape)}")
    finite(tensor, name)


def matching(tensor: torch.Tensor, reference: torch.Tensor, name: str,
             *, check_finite: bool = True) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.shape != reference.shape:
        raise ValueError(f"{name} must have shape {tuple(reference.shape)}")
    if tensor.dtype != reference.dtype or tensor.device != reference.device:
        raise ValueError(f"{name} must match the reference dtype and device")
    if check_finite:
        finite(tensor, name)


def scalar(value: float, name: str, *, minimum: float | None = None,
           maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
