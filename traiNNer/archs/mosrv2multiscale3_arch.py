from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from traiNNer.archs.arch_util import SampleMods3, UniUpsampleV3
from traiNNer.archs.mosrv2multiscale_arch import (
    MoSRv2MultiScale,
    _DeterministicNoiseInjection,
    _MoSRv2BlockStack,
)
from traiNNer.utils.registry import ARCH_REGISTRY


class _SequentialMoSRv2Refiner(nn.Module):
    def __init__(
        self,
        dim: int,
        num_blocks: int,
        expansion_ratio: float,
        rms_norm: bool,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(6, dim, 3, padding=1)
        self.blocks = _MoSRv2BlockStack(
            dim,
            num_blocks,
            expansion_ratio,
            rms_norm,
            gradient_checkpointing,
        )
        self.tail = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, padding=1),
            nn.Mish(inplace=True),
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.Mish(inplace=True),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.tail(self.blocks(self.stem(x)))


@ARCH_REGISTRY.register()
class MoSRv2MultiScale3(nn.Module):
    """Sequential restoration: MultiScale first, then MoSRv2 detail refinement."""

    def __init__(
        self,
        scale: int = 1,
        in_ch: int = 3,
        out_ch: int = 3,
        task: str = "descreen",
        encoder_dims: Sequence[int] = (16, 24, 40),
        encoder_blocks: Sequence[int] = (2, 4, 8),
        decoder_blocks: Sequence[int] = (4, 3, 2),
        num_downsamples: int = 2,
        context_blocks: int = 2,
        use_edge_refinement: bool = True,
        edge_refinement_blocks: int = 1,
        detail_dim: int = 32,
        detail_blocks: int = 16,
        expansion_ratio: float = 1.5,
        rms_norm: bool = False,
        noise_strength: float = 0.05,
        noise_layers: int = 2,
        upsampler: SampleMods3 = "pixelshuffledirect",
        upsampler_mid_dim: int = 64,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if task not in ("descreen", "noise"):
            raise ValueError("task must be 'descreen' or 'noise'.")
        if scale < 1:
            raise ValueError("MoSRv2MultiScale3 requires scale >= 1.")
        if in_ch != 3 or out_ch != 3:
            raise ValueError(
                "MoSRv2MultiScale3 requires exactly 3 input and output channels."
            )
        if detail_dim < 8:
            raise ValueError("detail_dim must be at least 8 for InceptionDWConv2d.")
        if detail_blocks < 0:
            raise ValueError("detail_blocks must be non-negative.")

        self.scale = scale
        self.task = task
        self.context_model = MoSRv2MultiScale(
            scale=1,
            in_ch=in_ch,
            out_ch=out_ch,
            task="descreen",
            encoder_dims=encoder_dims,
            encoder_blocks=encoder_blocks,
            decoder_blocks=decoder_blocks,
            num_downsamples=num_downsamples,
            context_blocks=context_blocks,
            expansion_ratio=expansion_ratio,
            rms_norm=rms_norm,
            use_edge_refinement=use_edge_refinement,
            edge_refinement_blocks=edge_refinement_blocks,
            noise_strength=noise_strength,
            noise_layers=noise_layers,
            upsampler=upsampler,
            upsampler_mid_dim=upsampler_mid_dim,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.detail_refiner = _SequentialMoSRv2Refiner(
            detail_dim,
            detail_blocks,
            expansion_ratio,
            rms_norm,
            gradient_checkpointing,
        )
        self.to_image = nn.Conv2d(detail_dim, out_ch, 3, padding=1)
        self.upsampler = (
            UniUpsampleV3(
                upsampler,
                scale,
                detail_dim,
                out_ch,
                upsampler_mid_dim,
            )
            if scale > 1
            else nn.Identity()
        )
        self.noise_injection = (
            _DeterministicNoiseInjection(
                detail_dim,
                out_ch,
                scale,
                noise_strength,
                noise_layers,
            )
            if task == "noise"
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                "MoSRv2MultiScale3 expects input with shape [B, 3, H, W]."
            )

        context_image = self.context_model(x)
        detail_features = self.detail_refiner(torch.cat((x, context_image), dim=1))
        if self.scale == 1:
            residual = self.to_image(detail_features)
            output_base = context_image
        else:
            residual = self.upsampler(detail_features)
            output_base = F.interpolate(
                context_image,
                scale_factor=self.scale,
                mode="bilinear",
                align_corners=False,
            )

        output = output_base + residual
        if self.task == "noise":
            output = output + self.noise_injection(detail_features)
        return output


__all__ = ["MoSRv2MultiScale3"]
