"""Scalar adaptive-KL learning-rate rule, applied once per optimization epoch."""

from ._validation import scalar


def adaptive_learning_rate(learning_rate: float, kl: float, *, threshold: float = 0.016,
                           min_lr: float = 1e-6, max_lr: float = 1e-2,
                           kl_factor: float = 2.0, lr_factor: float = 1.5) -> float:
    for name, value in (("learning_rate", learning_rate), ("threshold", threshold),
                        ("min_lr", min_lr), ("max_lr", max_lr)):
        scalar(value, name, minimum=0)
        if value == 0:
            raise ValueError(f"{name} must be positive")
    scalar(kl, "kl", minimum=0)
    scalar(kl_factor, "kl_factor", minimum=1)
    scalar(lr_factor, "lr_factor", minimum=1)
    if kl_factor == 1 or lr_factor == 1 or not min_lr <= learning_rate <= max_lr:
        raise ValueError("Factors must exceed 1 and learning rate must be within bounds")
    if kl > threshold * kl_factor:
        return max(min_lr, learning_rate / lr_factor)
    if kl < threshold / kl_factor:
        return min(max_lr, learning_rate * lr_factor)
    return learning_rate
