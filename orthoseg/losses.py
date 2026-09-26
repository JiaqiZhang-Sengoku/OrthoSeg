"""Paper-guided OrthoSeg objectives and alternating MID estimator.

The conditional Gaussian implements the paper's empirical CLUB-like log-ratio
objective. With a learned variational conditional distribution, this quantity
is an optimization surrogate; it is not automatically a certified MI bound.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConditionalGaussianMI(nn.Module):
    """Three-linear-layer q_phi(z_texture | z_structure), paper Eq. 1-2.

    Call ``auxiliary_loss`` for the q_phi optimizer step; its inputs are
    detached from the encoders. Then call ``mid_loss`` for the model optimizer
    step; it detaches q_phi weights while retaining gradients to both latent
    inputs. All off-diagonal batch pairs are used as negatives, with no random
    pairing or accidental positive pair in the marginal estimate.
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        if latent_dim < 1 or hidden_dim < 1:
            raise ValueError("latent_dim and hidden_dim must be positive")
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 2 * latent_dim)

    @staticmethod
    def _linear(x: Tensor, layer: nn.Linear, freeze: bool) -> Tensor:
        weight = layer.weight.detach() if freeze else layer.weight
        bias = layer.bias.detach() if freeze and layer.bias is not None else layer.bias
        return F.linear(x, weight, bias)

    def _parameters_for_structure(self, z_structure: Tensor, freeze: bool) -> tuple[Tensor, Tensor]:
        h = F.relu(self._linear(z_structure, self.fc1, freeze))
        h = F.relu(self._linear(h, self.fc2, freeze))
        mean, log_variance = self._linear(h, self.fc_out, freeze).chunk(2, dim=-1)
        # A diagonal Gaussian is an explicit implementation choice; clipping
        # log variance prevents exp over/underflow during alternating updates.
        return mean, log_variance.clamp(-10.0, 10.0)

    def log_prob_matrix(
        self, z_structure: Tensor, z_texture: Tensor, *, freeze_estimator: bool = False
    ) -> Tensor:
        """Return [N,N] log q_phi(z_texture[j] | z_structure[i])."""
        if (
            z_structure.ndim != 2
            or z_texture.ndim != 2
            or z_structure.shape != z_texture.shape
            or z_structure.shape[1] != self.latent_dim
        ):
            raise ValueError(f"expected matching [N, {self.latent_dim}] latent vectors")
        mean, log_variance = self._parameters_for_structure(z_structure, freeze_estimator)
        delta = z_texture.unsqueeze(0) - mean.unsqueeze(1)
        log_prob = -0.5 * (
            math.log(2.0 * math.pi)
            + log_variance.unsqueeze(1)
            + delta.square() * torch.exp(-log_variance).unsqueeze(1)
        )
        return log_prob.sum(dim=-1)

    def auxiliary_loss(self, z_structure: Tensor, z_texture: Tensor) -> Tensor:
        """Negative positive-pair log likelihood; update only q_phi with this."""
        log_prob = self.log_prob_matrix(z_structure.detach(), z_texture.detach())
        return -log_prob.diagonal().mean()

    def mid_loss(self, z_structure: Tensor, z_texture: Tensor) -> Tensor:
        """Positive mean minus all strictly off-diagonal negative mean.

        This loss needs at least two samples. q_phi parameters are detached,
        so backpropagation updates encoder/projection parameters only.
        """
        batch = z_structure.shape[0]
        if batch < 2:
            raise ValueError("MID needs batch size >= 2 for off-diagonal negatives")
        log_prob = self.log_prob_matrix(
            z_structure, z_texture, freeze_estimator=True
        )
        positive = log_prob.diagonal().mean()
        negative = (log_prob.sum() - log_prob.diagonal().sum()) / (batch * (batch - 1))
        return positive - negative


def mid_weight(
    epoch: int, *, total_epochs: int = 200, tau: float = 1.0, gamma: float = 0.9
) -> float:
    """Paper Eq. 9, with a zero-based epoch and values clamped after training."""
    if epoch < 0 or total_epochs < 1 or tau < 0 or gamma <= 0:
        raise ValueError("invalid MID schedule parameters")
    return tau * max(0.0, 1.0 - epoch / total_epochs) ** gamma


def _soft_dice_loss(probabilities: Tensor, targets: Tensor, smooth: float = 1.0) -> Tensor:
    probabilities = probabilities.flatten(start_dim=2)
    targets = targets.flatten(start_dim=2)
    intersection = (probabilities * targets).sum(dim=2)
    denominator = probabilities.sum(dim=2) + targets.sum(dim=2)
    return (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()


def segmentation_loss(logits: Tensor, targets: Tensor, *, task: str = "binary") -> Tensor:
    """Dice + cross entropy for binary, overlapping multi-label, or multiclass masks.

    Binary and multi-label targets are float tensors in [0,1] with the same
    [N,C,H,W] shape as logits. Multi-label uses independent sigmoid channels,
    which is required for nested optic disc/cup masks. Multiclass targets are
    integer class indices [N,H,W] or [N,1,H,W] and use softmax cross entropy.
    """
    if logits.ndim != 4:
        raise ValueError("logits must be [N, C, H, W]")
    if task in ("binary", "multilabel"):
        if task == "binary" and logits.shape[1] != 1:
            raise ValueError("binary task requires one output channel")
        if targets.shape != logits.shape:
            raise ValueError("binary/multilabel targets must match logits shape")
        targets = targets.to(dtype=logits.dtype)
        ce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = _soft_dice_loss(torch.sigmoid(logits), targets)
        return ce + dice
    if task == "multiclass":
        if logits.shape[1] < 2:
            raise ValueError("multiclass task requires at least two channels")
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]
        if targets.shape != (logits.shape[0], *logits.shape[-2:]):
            raise ValueError("multiclass targets must be [N, H, W]")
        targets = targets.long()
        ce = F.cross_entropy(logits, targets)
        one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
        dice = _soft_dice_loss(F.softmax(logits, dim=1), one_hot.to(logits.dtype))
        return ce + dice
    raise ValueError("task must be 'binary', 'multilabel', or 'multiclass'")


def structure_equivariance_loss(
    original_features: Sequence[Tensor],
    transformed_features: Sequence[Tensor],
    transform: Callable[[Tensor], Tensor],
) -> Tensor:
    """Mean across pyramid levels of ||P_s(Tx) - T(P_s(x))||_2^2.

    ``transform`` must apply the same geometry to a feature map at any scale.
    When it provides ``valid_mask(feature)``, padded pixels are excluded from
    the mean squared error. The caller is responsible for scale-aware grids.
    """
    if not original_features or len(original_features) != len(transformed_features):
        raise ValueError("feature pyramids must be nonempty and have equal lengths")
    losses = []
    for original, transformed in zip(original_features, transformed_features):
        expected = transform(original)
        if expected.shape != transformed.shape:
            raise ValueError("transformed feature shapes do not match geometric transform")
        valid_mask = getattr(transform, "valid_mask", None)
        if callable(valid_mask):
            mask = valid_mask(original)
            if mask.shape != (original.shape[0], 1, *original.shape[-2:]):
                raise ValueError("transform validity mask must be [N,1,H,W]")
            mask = mask.to(device=original.device, dtype=original.dtype)
            squared_error = (transformed - expected).square() * mask
            losses.append(squared_error.sum() / (mask.sum().clamp_min(1) * original.shape[1]))
        else:
            losses.append(F.mse_loss(transformed, expected))
    return torch.stack(losses).mean()


def texture_invariance_loss(original_latent: Tensor, transformed_latent: Tensor) -> Tensor:
    """||P_t(Tx) - P_t(x)||_2^2 on GAP-projected texture vectors.

    The paper does not specify whether Eq. 4 compares maps or pooled vectors.
    Pooled vectors encode its stated spatially agnostic texture objective and
    avoid penalizing the geometry of every raw texture feature pixel.
    """
    if original_latent.shape != transformed_latent.shape or original_latent.ndim != 2:
        raise ValueError("texture latents must have matching [N, D] shapes")
    return F.mse_loss(transformed_latent, original_latent)
