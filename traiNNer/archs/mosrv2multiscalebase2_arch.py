from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from traiNNer.archs.arch_util import SampleMods3, UniUpsampleV3
from traiNNer.archs.mosrv2multiscale_arch import (
    _DeterministicNoiseInjection,
    _Downsample,
    _MoSRv2BlockStack,
    _UpsampleWithSkip,
)
from traiNNer.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class MoSRv2MultiScaleBase2(nn.Module):
    """Shared RGB restoration backbone without panel-detection modes.

    ``forward_features`` exposes the source-resolution decoder features for
    future detail or pattern branches. Optional frequency guidance augments the
    stem with an input high-pass residual. Optional zero initialization makes
    RGB residual restoration start from an identity mapping.
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
        expansion_ratio: float = 1.5,
        rms_norm: bool = False,
        use_edge_refinement: bool = True,
        edge_refinement_blocks: int = 1,
        noise_strength: float = 0.05,
        noise_layers: int = 2,
        upsampler: SampleMods3 = "pixelshuffledirect",
        upsampler_mid_dim: int = 64,
        gradient_checkpointing: bool = False,
        zero_init_residual: bool = False,
        frequency_guidance: bool = False,
        frequency_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if task not in ("descreen", "noise"):
            raise ValueError("task must be 'descreen' or 'noise'.")
        if scale < 1:
            raise ValueError("MoSRv2MultiScaleBase2 requires scale >= 1.")
        if in_ch != 3 or out_ch != 3:
            raise ValueError(
                "MoSRv2MultiScaleBase2 requires exactly 3 input and output channels."
            )
        if num_downsamples < 0:
            raise ValueError("num_downsamples must be non-negative.")
        if frequency_kernel_size < 3 or frequency_kernel_size % 2 == 0:
            raise ValueError("frequency_kernel_size must be an odd integer >= 3.")

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
        if any(dim < 8 for dim in encoder_dims):
            raise ValueError(
                "All encoder_dims must be at least 8 for InceptionDWConv2d."
            )
        if any(blocks < 0 for blocks in (*encoder_blocks, *decoder_blocks)):
            raise ValueError("Block counts must be non-negative.")
        if context_blocks < 0 or edge_refinement_blocks < 0:
            raise ValueError(
                "context_blocks and edge_refinement_blocks must be non-negative."
            )
        if expansion_ratio <= 0:
            raise ValueError("expansion_ratio must be positive.")
        if noise_strength <= 0:
            raise ValueError("noise_strength must be positive.")
        if noise_layers < 1:
            raise ValueError("noise_layers must be at least 1.")

        dims = tuple(encoder_dims)
        self.task = task
        self.scale = scale
        self.num_downsamples = num_downsamples
        self.pad_factor = 2**num_downsamples
        self.feature_dim = dims[0]
        self.frequency_guidance = frequency_guidance
        self.lowpass = (
            nn.AvgPool2d(
                frequency_kernel_size,
                stride=1,
                padding=frequency_kernel_size // 2,
                count_include_pad=False,
            )
            if frequency_guidance
            else None
        )

        stem_in_ch = in_ch * (2 if frequency_guidance else 1)
        self.stem = nn.Conv2d(stem_in_ch, dims[0], 3, padding=1)
        self.encoder = nn.ModuleList(
            [
                _MoSRv2BlockStack(
                    dims[level],
                    encoder_blocks[level],
                    expansion_ratio,
                    rms_norm,
                    gradient_checkpointing,
                )
                for level in range(expected_levels)
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
        self.edge_refinement = (
            _MoSRv2BlockStack(
                dims[0],
                edge_refinement_blocks,
                expansion_ratio,
                rms_norm,
                gradient_checkpointing,
            )
            if use_edge_refinement
            else nn.Identity()
        )
        self.to_image = nn.Conv2d(dims[0], out_ch, 3, padding=1)
        if zero_init_residual:
            self._zero_init_output_layer(self.to_image)
        self.upsampler = (
            UniUpsampleV3(
                upsampler,
                scale,
                dims[0],
                out_ch,
                upsampler_mid_dim,
            )
            if scale > 1
            else nn.Identity()
        )
        if zero_init_residual and scale > 1:
            self._zero_init_output_layer(self.upsampler)
        self.noise_injection = (
            _DeterministicNoiseInjection(
                dims[0], out_ch, scale, noise_strength, noise_layers
            )
            if task == "noise"
            else nn.Identity()
        )

    @staticmethod
    def _zero_init_output_layer(module: nn.Module) -> None:
        for child in reversed(tuple(module.modules())):
            if isinstance(child, nn.Conv2d | nn.ConvTranspose2d):
                nn.init.zeros_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
                return
        raise ValueError("Expected an image-output convolution.")

    def _pad_input(self, x: Tensor) -> tuple[Tensor, int, int]:
        height, width = x.shape[-2:]
        pad_h = (self.pad_factor - height % self.pad_factor) % self.pad_factor
        pad_w = (self.pad_factor - width % self.pad_factor) % self.pad_factor
        if pad_h == 0 and pad_w == 0:
            return x, height, width
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), height, width

    def _forward_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                "MoSRv2MultiScaleBase2 expects input with shape [B, 3, H, W]."
            )

        x, height, width = self._pad_input(x)
        input_rgb = x
        if self.lowpass is not None:
            x = torch.cat((x, x - self.lowpass(x)), dim=1)
        x = self.stem(x)
        skips = []
        for level, encoder in enumerate(self.encoder):
            x = encoder(x)
            skips.append(x)
            if level < self.num_downsamples:
                x = self.downsamples[level](x)

        x = self.context(x)
        for decoder, skip in zip(self.decoder, reversed(skips[:-1]), strict=True):
            x = decoder(x, skip)

        refined = self.edge_refinement(x)
        if self.scale == 1:
            residual = self.to_image(refined)
            output_base = input_rgb
            noise_features = x
        else:
            residual = self.upsampler(refined)
            output_base = F.interpolate(
                input_rgb,
                scale_factor=self.scale,
                mode="bilinear",
                align_corners=False,
            )
            noise_features = x
        output = output_base + residual
        if self.task == "noise":
            output = output + self.noise_injection(noise_features)
        return (
            output[:, :, : height * self.scale, : width * self.scale],
            refined[:, :, :height, :width],
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_features(x)[0]

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return restored RGB plus source-resolution refined decoder features."""
        return self._forward_features(x)


__all__ = ["MoSRv2MultiScaleBase2"]
