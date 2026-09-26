"""Paper-guided, clean-room OrthoSeg reference implementation."""

from .losses import (
    ConditionalGaussianMI,
    mid_weight,
    segmentation_loss,
    structure_equivariance_loss,
    texture_invariance_loss,
)
from .model import CSTABlock, OrthoSeg, OrthoSegOutput

__all__ = [
    "CSTABlock",
    "ConditionalGaussianMI",
    "OrthoSeg",
    "OrthoSegOutput",
    "mid_weight",
    "segmentation_loss",
    "structure_equivariance_loss",
    "texture_invariance_loss",
]
