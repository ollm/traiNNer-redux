from collections.abc import Sequence
from typing import Literal

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


class _PatternOrientationAligner(nn.Module):
    """Predict and selectively rotate the input pattern residual."""

    def __init__(
        self,
        dim: int,
        target_angle_degrees: float,
        max_source_angle_degrees: float,
        padding_mode: Literal["zeros", "border", "reflection"],
    ) -> None:
        super().__init__()
        self.target_angle_radians = target_angle_degrees * torch.pi / 180
        self.max_source_angle_radians = max_source_angle_degrees * torch.pi / 180
        self.padding_mode = padding_mode
        self.features = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
        )
        self.source_angle_head = nn.Linear(dim, 1)
        self.source_angle = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(dim, dim),
            nn.GELU(),
            self.source_angle_head,
        )
        self.pattern_mask = nn.Sequential(nn.Conv2d(dim, 1, 1), nn.Sigmoid())
        nn.init.zeros_(self.source_angle_head.weight)
        nn.init.zeros_(self.source_angle_head.bias)

    def forward(self, source: Tensor, context: Tensor) -> Tensor:
        pattern_residual = source - context
        features = self.features(pattern_residual)
        source_angle = torch.tanh(self.source_angle(features).squeeze(1))
        source_angle = source_angle * self.max_source_angle_radians
        correction = self.target_angle_radians - source_angle

        cosine = torch.cos(correction)
        sine = torch.sin(correction)
        theta = torch.zeros(
            source.shape[0], 2, 3, dtype=source.dtype, device=source.device
        )
        # affine_grid maps output coordinates to input coordinates.
        theta[:, 0, 0] = cosine
        theta[:, 0, 1] = sine
        theta[:, 1, 0] = -sine
        theta[:, 1, 1] = cosine
        grid = F.affine_grid(theta, list(pattern_residual.shape), align_corners=False)
        rotated_pattern = F.grid_sample(
            pattern_residual,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=False,
        )
        mask = self.pattern_mask(features)
        return pattern_residual + mask * (rotated_pattern - pattern_residual)


@ARCH_REGISTRY.register()
class MoSRv2MultiScale4(nn.Module):
    """MultiScale restoration with optional selective learned pattern alignment.

    ``task`` accepts a comma-separated set of ``descreen``, ``noise``, and
    ``pattern``. ``noise`` enables deterministic texture injection; ``pattern``
    enables learned pattern alignment by default and can be combined with
    ``noise``. Set ``pattern_alignment_enabled`` explicitly to override that
    default. When pattern alignment is false, the computation is equivalent to
    :class:`MoSRv2MultiScale3`. When enabled, a learned module estimates the
    orientation of the input residual relative to the restored context, rotates
    that residual toward ``pattern_target_angle_degrees``, and blends it only
    where its learned mask identifies a pattern. The reoriented residual is
    composed directly with the context before final detail refinement.
    """

    def __init__(  # noqa: PLR0917
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
        pattern_alignment_enabled: bool | None = None,
        pattern_alignment_dim: int = 16,
        pattern_target_angle_degrees: float = 30.0,
        pattern_max_source_angle_degrees: float = 180.0,
        pattern_alignment_padding_mode: Literal[
            "zeros", "border", "reflection"
        ] = "border",
    ) -> None:
        super().__init__()
        task_names = tuple(name.strip() for name in task.split(",") if name.strip())
        supported_tasks = frozenset(("descreen", "noise", "pattern"))
        if (
            not task_names
            or len(task_names) != len(set(task_names))
            or not set(task_names).issubset(supported_tasks)
        ):
            raise ValueError(
                "task must be a comma-separated combination of 'descreen', "
                "'noise', and 'pattern'."
            )
        if scale < 1:
            raise ValueError("MoSRv2MultiScale4 requires scale >= 1.")
        if in_ch != 3 or out_ch != 3:
            raise ValueError(
                "MoSRv2MultiScale4 requires exactly 3 input and output channels."
            )
        if detail_dim < 8:
            raise ValueError("detail_dim must be at least 8 for InceptionDWConv2d.")
        if detail_blocks < 0:
            raise ValueError("detail_blocks must be non-negative.")
        if pattern_alignment_dim < 1:
            raise ValueError("pattern_alignment_dim must be positive.")
        if not -180 <= pattern_target_angle_degrees <= 180:
            raise ValueError("pattern_target_angle_degrees must be in [-180, 180].")
        if not 0 < pattern_max_source_angle_degrees <= 180:
            raise ValueError("pattern_max_source_angle_degrees must be in (0, 180].")

        self.scale = scale
        self.task = task
        self.tasks = frozenset(task_names)
        if pattern_alignment_enabled is None:
            pattern_alignment_enabled = "pattern" in self.tasks
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
        self.pattern_aligner = (
            _PatternOrientationAligner(
                pattern_alignment_dim,
                pattern_target_angle_degrees,
                pattern_max_source_angle_degrees,
                pattern_alignment_padding_mode,
            )
            if pattern_alignment_enabled
            else None
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
            if "noise" in self.tasks
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("MoSRv2MultiScale4 expects input with shape [B, 3, H, W].")

        context_image = self.context_model(x)
        if self.pattern_aligner is None:
            detail_input = x
            output_base = context_image
        else:
            aligned_pattern = self.pattern_aligner(x, context_image)
            output_base = context_image + aligned_pattern
            detail_input = output_base
        detail_features = self.detail_refiner(
            torch.cat((detail_input, context_image), dim=1)
        )
        if self.scale == 1:
            residual = self.to_image(detail_features)
        else:
            residual = self.upsampler(detail_features)
            output_base = F.interpolate(
                output_base,
                scale_factor=self.scale,
                mode="bilinear",
                align_corners=False,
            )

        output = output_base + residual
        if "noise" in self.tasks:
            output = output + self.noise_injection(detail_features)
        return output


__all__ = ["MoSRv2MultiScale4"]
