from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from traiNNer.archs.mosrv2_arch import GatedCNNBlock
from traiNNer.utils.registry import ARCH_REGISTRY


class _BlockStack(nn.Module):
    def __init__(
        self,
        dim: int,
        num_blocks: int,
        expansion_ratio: float,
        rms_norm: bool,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                GatedCNNBlock(
                    dim=dim,
                    expansion_ratio=expansion_ratio,
                    rms_norm=rms_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class _StrideTwoConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _UpsampleWithSkip(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_blocks: int,
        expansion_ratio: float,
        rms_norm: bool,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.up = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.skip = nn.Conv2d(out_dim, out_dim, 1)
        self.blocks = _BlockStack(
            out_dim,
            num_blocks,
            expansion_ratio,
            rms_norm,
            gradient_checkpointing,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.blocks(self.up(x) + self.skip(skip))


class _MoSRv2DetailBranch(nn.Module):
    """MoSRv2 full-resolution feature trunk without its SR image head."""

    def __init__(
        self,
        in_ch: int,
        dim: int,
        num_blocks: int,
        expansion_ratio: float,
        rms_norm: bool,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_ch, dim, 3, padding=1)
        self.blocks = _BlockStack(
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


class _GlobalContext(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return features + self.project(self.pool(features))


class _ContextBranch(nn.Module):
    def __init__(
        self,
        in_ch: int,
        dims: Sequence[int],
        encoder_blocks: Sequence[int],
        decoder_blocks: Sequence[int],
        context_blocks: int,
        out_dim: int,
        expansion_ratio: float,
        rms_norm: bool,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_ch, dims[0], 3, padding=1)
        self.encoder = nn.ModuleList(
            [
                _BlockStack(
                    dims[level],
                    encoder_blocks[level],
                    expansion_ratio,
                    rms_norm,
                    gradient_checkpointing,
                )
                for level in range(len(dims))
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                _StrideTwoConv(dims[level], dims[level + 1])
                for level in range(len(dims) - 1)
            ]
        )
        self.global_context = _GlobalContext(dims[-1])
        self.context = _BlockStack(
            dims[-1],
            context_blocks,
            expansion_ratio,
            rms_norm,
            gradient_checkpointing,
        )
        self.decoder = nn.ModuleList(
            [
                _UpsampleWithSkip(
                    dims[level + 1],
                    dims[level],
                    decoder_blocks[len(dims) - 2 - level],
                    expansion_ratio,
                    rms_norm,
                    gradient_checkpointing,
                )
                for level in range(len(dims) - 2, -1, -1)
            ]
        )
        self.project = nn.Conv2d(dims[0], out_dim, 1)

    def forward(self, x: Tensor, output_size: tuple[int, int]) -> Tensor:
        x = self.stem(x)
        skips = []
        for level, encoder in enumerate(self.encoder):
            x = encoder(x)
            skips.append(x)
            if level < len(self.downsamples):
                x = self.downsamples[level](x)
        x = self.context(self.global_context(x))
        for decoder, skip in zip(
            self.decoder, reversed(skips[:-1]), strict=True
        ):
            x = decoder(x, skip)
        x = self.project(x)
        return x[:, :, : output_size[0], : output_size[1]]


@ARCH_REGISTRY.register()
class MoSRv2Panels4(nn.Module):
    """Full-resolution MoSRv2 edge features fused with global panel context.

    The detail trunk follows MoSRv2's GatedCNNBlock stack and convolutional tail.
    A parallel encoder supplies both spatially coarse and pooled global context.
    Output R/B are copied from the input and G contains mask logits.
    """

    def __init__(
        self,
        scale: int = 1,
        detail_dim: int = 32,
        detail_blocks: int = 16,
        context_dims: Sequence[int] = (16, 24, 32, 48),
        context_encoder_blocks: Sequence[int] = (2, 2, 4, 6),
        context_decoder_blocks: Sequence[int] | None = None,
        context_blocks: int = 2,
        fusion_blocks: int = 1,
        mask_head_blocks: int = 2,
        expansion_ratio: float = 1.5,
        rms_norm: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if scale != 1:
            raise ValueError("MoSRv2Panels4 requires scale=1.")
        if detail_dim < 8:
            raise ValueError("detail_dim must be at least 8 for InceptionDWConv2d.")
        if detail_blocks < 0:
            raise ValueError("detail_blocks must be non-negative.")
        if not context_dims:
            raise ValueError("context_dims must contain at least one level.")
        if len(context_dims) != len(context_encoder_blocks):
            raise ValueError(
                "context_dims and context_encoder_blocks must have equal lengths."
            )
        if context_decoder_blocks is None:
            context_decoder_blocks = (2,) * (len(context_dims) - 1)
        if len(context_decoder_blocks) != len(context_dims) - 1:
            raise ValueError(
                "context_decoder_blocks must contain len(context_dims) - 1 values."
            )
        if any(dim < 8 for dim in context_dims):
            raise ValueError(
                "All context_dims must be at least 8 for InceptionDWConv2d."
            )
        if any(
            blocks < 0
            for blocks in (*context_encoder_blocks, *context_decoder_blocks)
        ):
            raise ValueError("Context encoder and decoder block counts must be non-negative.")
        if context_blocks < 0 or fusion_blocks < 0 or mask_head_blocks < 0:
            raise ValueError("Context, fusion, and mask head block counts must be non-negative.")
        if expansion_ratio <= 0:
            raise ValueError("expansion_ratio must be positive.")

        self.context_pad_factor = 2 ** (len(context_dims) - 1)
        self.detail_branch = _MoSRv2DetailBranch(
            in_ch=2,
            dim=detail_dim,
            num_blocks=detail_blocks,
            expansion_ratio=expansion_ratio,
            rms_norm=rms_norm,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.context_branch = _ContextBranch(
            in_ch=2,
            dims=context_dims,
            encoder_blocks=context_encoder_blocks,
            decoder_blocks=context_decoder_blocks,
            context_blocks=context_blocks,
            out_dim=detail_dim,
            expansion_ratio=expansion_ratio,
            rms_norm=rms_norm,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(detail_dim * 2, detail_dim, 1),
            nn.GELU(),
        )
        self.fusion_refinement = _BlockStack(
            detail_dim,
            fusion_blocks,
            expansion_ratio,
            rms_norm,
            gradient_checkpointing,
        )
        mask_layers: list[nn.Module] = []
        for _ in range(mask_head_blocks):
            mask_layers.extend(
                (nn.Conv2d(detail_dim, detail_dim, 3, padding=1), nn.GELU())
            )
        mask_layers.append(nn.Conv2d(detail_dim, 1, 1))
        self.to_mask = nn.Sequential(*mask_layers)

    def _pad_input(self, x: Tensor) -> tuple[Tensor, int, int]:
        height, width = x.shape[-2:]
        pad_h = (self.context_pad_factor - height % self.context_pad_factor) % self.context_pad_factor
        pad_w = (self.context_pad_factor - width % self.context_pad_factor) % self.context_pad_factor
        if pad_h == 0 and pad_w == 0:
            return x, height, width
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), height, width

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("MoSRv2Panels4 expects input with shape [B, 3, H, W].")

        x, height, width = self._pad_input(x)
        input_rgb = x
        panel_channels = torch.cat((x[:, 0:1], x[:, 2:3]), dim=1)
        detail = self.detail_branch(panel_channels)
        context = self.context_branch(panel_channels, detail.shape[-2:])
        fused = detail + self.fuse(torch.cat((detail, context), dim=1))
        fused = self.fusion_refinement(fused)
        mask_logits = self.to_mask(fused)
        output = torch.cat(
            (input_rgb[:, 0:1], mask_logits, input_rgb[:, 2:3]), dim=1
        )
        return output[:, :, :height, :width]


__all__ = ["MoSRv2Panels4"]
