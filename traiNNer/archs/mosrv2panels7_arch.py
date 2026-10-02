from collections.abc import Sequence

import torch
from torch import Tensor, nn

from traiNNer.archs.mosrv2panels3_arch import MoSRv2Panels3
from traiNNer.archs.mosrv2panels5_arch import _SequentialMaskRefiner
from traiNNer.utils.registry import ARCH_REGISTRY


class _CoordinateMoSRv2Panels3(MoSRv2Panels3):
    """Panels3 detector that receives R/B plus normalized pixel coordinates."""

    def __init__(  # noqa: PLR0917
        self,
        scale: int = 1,
        encoder_dims: Sequence[int] = (16, 24, 40),
        encoder_blocks: Sequence[int] = (2, 4, 8),
        decoder_blocks: Sequence[int] = (4, 3, 2),
        num_downsamples: int = 2,
        context_blocks: int = 2,
        expansion_ratio: float = 1.5,
        rms_norm: bool = False,
        mask_head_blocks: int = 2,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(
            scale=scale,
            encoder_dims=encoder_dims,
            encoder_blocks=encoder_blocks,
            decoder_blocks=decoder_blocks,
            num_downsamples=num_downsamples,
            context_blocks=context_blocks,
            expansion_ratio=expansion_ratio,
            rms_norm=rms_norm,
            mask_head_blocks=mask_head_blocks,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.stem = nn.Conv2d(4, encoder_dims[0], 3, padding=1)

    @staticmethod
    def _coordinate_channels(x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        y_coords = torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype)
        x_coords = torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=0).expand(batch, -1, -1, -1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                "_CoordinateMoSRv2Panels3 expects input with shape [B, 3, H, W]."
            )

        x, height, width = self._pad_input(x)
        input_rgb = x
        panel_channels = torch.cat(
            (x[:, 0:1], x[:, 2:3], self._coordinate_channels(x)), dim=1
        )
        features = self.stem(panel_channels)
        skips = []
        for level, encoder in enumerate(self.encoder):
            features = encoder(features)
            skips.append(features)
            if level < self.num_downsamples:
                features = self.downsamples[level](features)

        features = self.context(self.global_context(features))
        for decoder, skip in zip(
            self.decoder, reversed(skips[:-1]), strict=True
        ):
            features = decoder(features, skip)

        mask_logits = self.to_mask(features)
        output = torch.cat(
            (input_rgb[:, 0:1], mask_logits, input_rgb[:, 2:3]), dim=1
        )
        return output[:, :, :height, :width]


@ARCH_REGISTRY.register()
class MoSRv2Panels7(nn.Module):
    """Coordinate-aware Panels3 U-Net followed by residual mask refinement.

    The first stage ignores input G and detects panels from R/B plus normalized
    horizontal and vertical pixel coordinates. Its U-Net uses global context and
    skip connections; the second stage refines its mask logits at full resolution.
    """

    def __init__(  # noqa: PLR0917
        self,
        scale: int = 1,
        encoder_dims: Sequence[int] = (16, 24, 40),
        encoder_blocks: Sequence[int] = (2, 4, 8),
        decoder_blocks: Sequence[int] = (4, 3, 2),
        num_downsamples: int = 2,
        context_blocks: int = 2,
        mask_head_blocks: int = 2,
        detail_dim: int = 32,
        detail_blocks: int = 16,
        expansion_ratio: float = 1.5,
        rms_norm: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if scale != 1:
            raise ValueError("MoSRv2Panels7 requires scale=1.")
        if detail_dim < 8:
            raise ValueError("detail_dim must be at least 8 for InceptionDWConv2d.")
        if detail_blocks < 0 or mask_head_blocks < 0:
            raise ValueError("detail_blocks and mask_head_blocks must be non-negative.")

        self.context_model = _CoordinateMoSRv2Panels3(
            scale=1,
            encoder_dims=encoder_dims,
            encoder_blocks=encoder_blocks,
            decoder_blocks=decoder_blocks,
            num_downsamples=num_downsamples,
            context_blocks=context_blocks,
            expansion_ratio=expansion_ratio,
            rms_norm=rms_norm,
            mask_head_blocks=mask_head_blocks,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.detail_refiner = _SequentialMaskRefiner(
            detail_dim,
            detail_blocks,
            mask_head_blocks,
            expansion_ratio,
            rms_norm,
            gradient_checkpointing,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("MoSRv2Panels7 expects input with shape [B, 3, H, W].")

        context_output = self.context_model(x)
        refinement_input = torch.cat((x, context_output), dim=1)
        mask_logits = context_output[:, 1:2] + self.detail_refiner(refinement_input)
        return torch.cat((x[:, 0:1], mask_logits, x[:, 2:3]), dim=1)


__all__ = ["MoSRv2Panels7"]
