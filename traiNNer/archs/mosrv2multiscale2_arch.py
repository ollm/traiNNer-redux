from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from traiNNer.archs.arch_util import SampleMods3, UniUpsampleV3
from traiNNer.archs.mosrv2_arch import GatedCNNBlock
from traiNNer.archs.mosrv2multiscale_arch import (
    _DeterministicNoiseInjection,
    _Downsample,
    _MoSRv2BlockStack,
    _UpsampleWithSkip,
)
from traiNNer.utils.registry import ARCH_REGISTRY


class _MoSRv2DetailBranch(nn.Module):
    def __init__(
        self,
        dim: int,
        num_blocks: int,
        expansion_ratio: float,
        rms_norm: bool,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, dim, 3, padding=1)
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


class _MultiScaleContextBranch(nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        encoder_blocks: Sequence[int],
        decoder_blocks: Sequence[int],
        num_downsamples: int,
        context_blocks: int,
        expansion_ratio: float,
        rms_norm: bool,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, dims[0], 3, padding=1)
        self.encoder = nn.ModuleList(
            [
                _MoSRv2BlockStack(
                    dims[level],
                    encoder_blocks[level],
                    expansion_ratio,
                    rms_norm,
                    gradient_checkpointing,
                )
                for level in range(num_downsamples + 1)
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                _Downsample(dims[level], dims[level + 1])
                for level in range(num_downsamples)
            ]
        )
        self.context = _MoSRv2BlockStack(
            dims[-1],
            context_blocks + decoder_blocks[0],
            expansion_ratio,
            rms_norm,
            gradient_checkpointing,
        )
        self.decoder = nn.ModuleList(
            [
                _UpsampleWithSkip(
                    dims[level + 1],
                    dims[level],
                    decoder_blocks[num_downsamples - level],
                    expansion_ratio,
                    rms_norm,
                    gradient_checkpointing,
                )
                for level in range(num_downsamples - 1, -1, -1)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        skips = []
        for level, encoder in enumerate(self.encoder):
            x = encoder(x)
            skips.append(x)
            if level < len(self.downsamples):
                x = self.downsamples[level](x)
        x = self.context(x)
        for decoder, skip in zip(self.decoder, reversed(skips[:-1]), strict=True):
            x = decoder(x, skip)
        return x


@ARCH_REGISTRY.register()
class MoSRv2MultiScale2(nn.Module):
    """Restoration with parallel multiscale-context and MoSRv2 detail branches.

    ``task='descreen'`` predicts the restoration/upscaling residual. ``task='noise'``
    additionally applies deterministic, feature-conditioned texture after reconstruction.
    """

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
        detail_dim: int = 32,
        detail_blocks: int = 16,
        fusion_blocks: int = 1,
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
            raise ValueError("MoSRv2MultiScale2 requires scale >= 1.")
        if in_ch != 3 or out_ch != 3:
            raise ValueError(
                "MoSRv2MultiScale2 requires exactly 3 input and output channels."
            )
        if num_downsamples < 0:
            raise ValueError("num_downsamples must be non-negative.")

        expected_levels = num_downsamples + 1
        for name, values in (
            ("encoder_dims", encoder_dims),
            ("encoder_blocks", encoder_blocks),
            ("decoder_blocks", decoder_blocks),
        ):
            if len(values) != expected_levels:
                raise ValueError(
                    f"{name} must contain num_downsamples + 1 values: "
                    f"expected {expected_levels}, got {len(values)}."
                )
        if any(dim < 8 for dim in (*encoder_dims, detail_dim)):
            raise ValueError("All feature dimensions must be at least 8.")
        if detail_blocks < 0 or fusion_blocks < 0:
            raise ValueError("detail_blocks and fusion_blocks must be non-negative.")
        if any(blocks < 0 for blocks in (*encoder_blocks, *decoder_blocks)):
            raise ValueError("Encoder and decoder block counts must be non-negative.")
        if context_blocks < 0:
            raise ValueError("context_blocks must be non-negative.")
        if expansion_ratio <= 0:
            raise ValueError("expansion_ratio must be positive.")
        if noise_strength <= 0:
            raise ValueError("noise_strength must be positive.")
        if noise_layers < 1:
            raise ValueError("noise_layers must be at least 1.")

        dims = tuple(encoder_dims)
        self.scale = scale
        self.num_downsamples = num_downsamples
        self.pad_factor = 2**num_downsamples
        self.context_branch = _MultiScaleContextBranch(
            dims,
            encoder_blocks,
            decoder_blocks,
            num_downsamples,
            context_blocks,
            expansion_ratio,
            rms_norm,
            gradient_checkpointing,
        )
        self.detail_branch = _MoSRv2DetailBranch(
            detail_dim,
            detail_blocks,
            expansion_ratio,
            rms_norm,
            gradient_checkpointing,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(dims[0] + detail_dim, detail_dim, 1),
            nn.GELU(),
        )
        self.fusion_refinement = _MoSRv2BlockStack(
            detail_dim,
            fusion_blocks,
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
        self.task = task

    def _pad_input(self, x: Tensor) -> tuple[Tensor, int, int]:
        height, width = x.shape[-2:]
        pad_h = (self.pad_factor - height % self.pad_factor) % self.pad_factor
        pad_w = (self.pad_factor - width % self.pad_factor) % self.pad_factor
        if pad_h == 0 and pad_w == 0:
            return x, height, width
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), height, width

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                "MoSRv2MultiScale2 expects input with shape [B, 3, H, W]."
            )

        x, height, width = self._pad_input(x)
        context_features = self.context_branch(x)
        detail_features = self.detail_branch(x)
        fused = detail_features + self.fuse(
            torch.cat((context_features, detail_features), dim=1)
        )
        fused = self.fusion_refinement(fused)

        if self.scale == 1:
            residual = self.to_image(fused)
            output_base = x
        else:
            residual = self.upsampler(fused)
            output_base = F.interpolate(
                x,
                scale_factor=self.scale,
                mode="bilinear",
                align_corners=False,
            )

        output = output_base + residual
        if self.task == "noise":
            output = output + self.noise_injection(fused)
        return output[:, :, : height * self.scale, : width * self.scale]


__all__ = ["MoSRv2MultiScale2"]
