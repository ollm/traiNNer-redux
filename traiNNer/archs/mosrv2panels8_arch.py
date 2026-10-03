from collections.abc import Sequence

import torch
from torch import Tensor, nn

from traiNNer.archs.mosrv2panels5_arch import _SequentialMaskRefiner
from traiNNer.archs.mosrv2panels7_arch import _CoordinateMoSRv2Panels3
from traiNNer.utils.registry import ARCH_REGISTRY


class _ThreeChannelMaskRefiner(_SequentialMaskRefiner):
    """Residual mask refiner fed by source R/B and initial mask logits only."""

    def __init__(  # noqa: PLR0917
        self,
        dim: int,
        num_blocks: int,
        mask_head_blocks: int,
        expansion_ratio: float,
        rms_norm: bool,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__(
            dim,
            num_blocks,
            mask_head_blocks,
            expansion_ratio,
            rms_norm,
            gradient_checkpointing,
        )
        self.stem = nn.Conv2d(3, dim, 3, padding=1)


@ARCH_REGISTRY.register()
class MoSRv2Panels8(nn.Module):
    """Coordinate-aware panel detector with an R/B plus initial-mask refiner.

    The first stage predicts panel logits from source R/B and normalized pixel
    coordinates. The refiner receives the original R/B samples and first-stage
    mask logits, avoiding the constant input G and duplicated R/B channels.
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
            raise ValueError("MoSRv2Panels8 requires scale=1.")
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
        self.detail_refiner = _ThreeChannelMaskRefiner(
            detail_dim,
            detail_blocks,
            mask_head_blocks,
            expansion_ratio,
            rms_norm,
            gradient_checkpointing,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("MoSRv2Panels8 expects input with shape [B, 3, H, W].")

        context_output = self.context_model(x)
        refinement_input = torch.cat(
            (x[:, 0:1], x[:, 2:3], context_output[:, 1:2]), dim=1
        )
        mask_logits = context_output[:, 1:2] + self.detail_refiner(refinement_input)
        return torch.cat((x[:, 0:1], mask_logits, x[:, 2:3]), dim=1)


__all__ = ["MoSRv2Panels8"]
