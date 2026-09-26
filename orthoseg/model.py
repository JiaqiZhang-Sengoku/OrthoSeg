"""A runnable, paper-guided reference implementation of OrthoSeg.

The paper specifies the dual independent encoders, CSTA routing statistics,
four dynamic kernel experts, and structure-only decoder. It does not specify
the exact encoder block widths, normalization, decoder blocks, or convolution
kernel dimensions. Those engineering choices are explicit here; this module
must not be treated as the authors' original checkpoint-compatible model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    """Choose a GroupNorm grouping that also works for a 1x1 feature map."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )


class PyramidEncoder(nn.Module):
    """Four-scale CNN pyramid; structure and texture instances share no weights."""

    def __init__(self, in_channels: int, widths: Tuple[int, ...]) -> None:
        super().__init__()
        stages = []
        previous = in_channels
        for index, width in enumerate(widths):
            stages.append(
                nn.Sequential(
                    ConvNormAct(previous, width, stride=1 if index == 0 else 2),
                    ConvNormAct(width, width),
                )
            )
            previous = width
        self.stages = nn.ModuleList(stages)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return tuple(features)


class CSTABlock(nn.Module):
    """Mean/std-routed per-sample convolution, corresponding to paper Eq. 5-7.

    Each expert is an ordinary 3x3 convolutional kernel. The paper leaves the
    expert kernel size, bias, and post-convolution normalization unspecified;
    this reference uses no expert bias followed by GroupNorm and ReLU.
    """

    def __init__(self, channels: int, num_experts: int = 4, kernel_size: int = 3) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError("CSTA channels must be at least 2")
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.channels = channels
        self.num_experts = num_experts
        self.kernel_size = kernel_size
        self.experts = nn.Parameter(
            torch.empty(num_experts, channels, channels, kernel_size, kernel_size)
        )
        hidden = max(8, channels // 4)
        self.router = nn.Sequential(
            nn.Linear(2 * channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_experts),
        )
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.activation = nn.ReLU(inplace=True)
        for expert in self.experts:
            nn.init.kaiming_normal_(expert, mode="fan_out", nonlinearity="relu")

    def routing_weights(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected [N, {self.channels}, H, W] feature map")
        mean = x.mean(dim=(-2, -1))
        # Eq. 5 uses population standard deviation, not Bessel correction.
        std = x.std(dim=(-2, -1), unbiased=False)
        descriptor = torch.cat((mean, std), dim=1)
        return F.softmax(self.router(descriptor), dim=1)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        routing = self.routing_weights(x)
        # Build one complete convolution kernel per image. Grouping by batch
        # evaluates all samples in a single conv2d call without a Python loop.
        kernels = torch.einsum("nk,kocij->nocij", routing, self.experts)
        merged_input = x.reshape(1, batch * channels, height, width)
        merged_kernels = kernels.reshape(
            batch * channels, channels, self.kernel_size, self.kernel_size
        )
        y = F.conv2d(
            merged_input,
            merged_kernels,
            padding=self.kernel_size // 2,
            groups=batch,
        )
        y = y.reshape(batch, channels, height, width)
        return self.activation(self.norm(y))


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels),
            ConvNormAct(out_channels, out_channels),
        )

    def forward(self, lower: Tensor, skip: Tensor) -> Tensor:
        # Explicit target size handles odd image dimensions without cropping.
        lower = F.interpolate(lower, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((lower, skip), dim=1))


@dataclass(frozen=True)
class OrthoSegOutput:
    """Extra training representations; logits are unnormalized segmentation scores."""

    logits: Tensor
    structure_features: Tuple[Tensor, ...]
    structure_latent: Tensor
    texture_latent: Tensor


class OrthoSeg(nn.Module):
    """Dual-encoder OrthoSeg reference model.

    ``widths`` controls the four encoder/decoder pyramid widths. The default
    is deliberately light for reproducible CPU smoke tests. Using larger
    widths changes capacity but does not make the unspecified paper blocks
    checkpoint-compatible with the author's model.

    ``forward(x)`` returns segmentation logits. For training, call
    ``forward(x, return_features=True)`` to obtain the latent vectors needed
    for MID and GCC. Texture encoding is skipped at pure inference because it
    is not consumed by the structure-only segmentation decoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        widths: Tuple[int, int, int, int] = (16, 32, 64, 128),
        latent_dim: int = 256,
        num_experts: int = 4,
    ) -> None:
        super().__init__()
        if in_channels < 1 or num_classes < 1 or latent_dim < 1:
            raise ValueError("in_channels, num_classes and latent_dim must be positive")
        if len(widths) != 4 or any(width < 2 for width in widths):
            raise ValueError("widths must contain four integers of at least 2")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.widths = tuple(widths)
        self.latent_dim = latent_dim

        self.structure_encoder = PyramidEncoder(in_channels, self.widths)
        self.texture_encoder = PyramidEncoder(in_channels, self.widths)
        self.structure_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(self.widths[-1], latent_dim)
        )
        self.texture_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(self.widths[-1], latent_dim)
        )
        self.csta = nn.ModuleList(
            CSTABlock(width, num_experts=num_experts) for width in self.widths
        )
        self.decoder = nn.ModuleList(
            _DecoderBlock(self.widths[level + 1] + self.widths[level], self.widths[level])
            for level in range(len(self.widths) - 2, -1, -1)
        )
        self.head = nn.Conv2d(self.widths[0], num_classes, kernel_size=1)

    def forward(
        self, x: Tensor, return_features: bool = False
    ) -> Union[Tensor, OrthoSegOutput]:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected input [N, {self.in_channels}, H, W]")
        if x.shape[0] < 1 or min(x.shape[-2:]) < 1:
            raise ValueError("batch and spatial dimensions must be positive")

        structure_features = self.structure_encoder(x)
        adapted = [block(feature) for block, feature in zip(self.csta, structure_features)]
        decoded = adapted[-1]
        for decoder, skip in zip(self.decoder, reversed(adapted[:-1])):
            decoded = decoder(decoded, skip)
        logits = self.head(decoded)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if not return_features:
            return logits

        texture_features = self.texture_encoder(x)
        return OrthoSegOutput(
            logits=logits,
            structure_features=structure_features,
            structure_latent=self.structure_projection(structure_features[-1]),
            texture_latent=self.texture_projection(texture_features[-1]),
        )
