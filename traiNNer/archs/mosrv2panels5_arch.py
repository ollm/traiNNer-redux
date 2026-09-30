from collections.abc import Sequence

import torch
from torch import Tensor, nn

from traiNNer.archs.mosrv2multiscale_arch import (
    MoSRv2MultiScale,
    _MoSRv2BlockStack,
)
from traiNNer.utils.registry import ARCH_REGISTRY


class _SequentialMaskRefiner(nn.Module):
    def __init__(
        self,
        dim: int,
        num_blocks: int,
        mask_head_blocks: int,
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
        mask_layers: list[nn.Module] = []
        for _ in range(mask_head_blocks):
            mask_layers.extend((nn.Conv2d(dim, dim, 3, padding=1), nn.GELU()))
        mask_layers.append(nn.Conv2d(dim, 1, 3, padding=1))
        self.to_mask_residual = nn.Sequential(*mask_layers)
        nn.init.zeros_(self.to_mask_residual[-1].weight)
        nn.init.zeros_(self.to_mask_residual[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        features = self.tail(self.blocks(self.stem(x)))
        return self.to_mask_residual(features)


@ARCH_REGISTRY.register()
class MoSRv2Panels5(nn.Module):
    """Sequential panel masks: MultiScale predicts logits, MoSRv2 refines them."""

    def __init__(
        self,
        scale: int = 1,
        encoder_dims: Sequence[int] = (16, 24, 40),
        encoder_blocks: Sequence[int] = (2, 4, 8),
        decoder_blocks: Sequence[int] = (4, 3, 2),
        num_downsamples: int = 2,
        context_blocks: int = 2,
        use_edge_refinement: bool = True,
        edge_refinement_blocks: int = 1,
        mask_head_blocks: int = 2,
        detail_dim: int = 32,
        detail_blocks: int = 16,
        expansion_ratio: float = 1.5,
        rms_norm: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if scale != 1:
            raise ValueError("MoSRv2Panels5 requires scale=1.")
        if detail_dim < 8:
            raise ValueError("detail_dim must be at least 8 for InceptionDWConv2d.")
        if detail_blocks < 0 or mask_head_blocks < 0:
            raise ValueError("detail_blocks and mask_head_blocks must be non-negative.")

        self.context_model = MoSRv2MultiScale(
            scale=1,
            task="panels2",
            encoder_dims=encoder_dims,
            encoder_blocks=encoder_blocks,
            decoder_blocks=decoder_blocks,
            num_downsamples=num_downsamples,
            context_blocks=context_blocks,
            expansion_ratio=expansion_ratio,
            rms_norm=rms_norm,
            use_edge_refinement=use_edge_refinement,
            edge_refinement_blocks=edge_refinement_blocks,
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
            raise ValueError("MoSRv2Panels5 expects input with shape [B, 3, H, W].")

        context_output = self.context_model(x)
        refinement_input = torch.cat((x, context_output), dim=1)
        mask_logits = context_output[:, 1:2] + self.detail_refiner(refinement_input)
        return torch.cat((x[:, 0:1], mask_logits, x[:, 2:3]), dim=1)


__all__ = ["MoSRv2Panels5"]
