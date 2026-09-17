from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from traiNNer.archs.mosrv2_arch import GatedCNNBlock
from traiNNer.utils.registry import ARCH_REGISTRY


class _MoSRv2BlockStack(nn.Module):
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


class _Downsample(nn.Module):
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
        self.blocks = _MoSRv2BlockStack(
            out_dim,
            num_blocks,
            expansion_ratio,
            rms_norm,
            gradient_checkpointing,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.blocks(self.up(x) + self.skip(skip))


@ARCH_REGISTRY.register()
class MoSRv2Panels(nn.Module):
    """Dynamic-resolution MoSRv2 encoder-decoder for panel masks.

    The network returns three unbounded channels at the input resolution. The
    green channel is intentionally left as a logit for BCE-with-logits losses;
    red and blue use residual image reconstruction from the input.
    """

    def __init__(
        self,
        scale: int = 1,
        in_ch: int = 3,
        out_ch: int = 3,
        encoder_dims: Sequence[int] = (16, 24, 40),
        encoder_blocks: Sequence[int] = (2, 4, 8),
        decoder_blocks: Sequence[int] = (4, 3, 2),
        num_downsamples: int = 2,
        context_blocks: int = 2,
        expansion_ratio: float = 1.5,
        rms_norm: bool = False,
        use_edge_refinement: bool = True,
        edge_refinement_blocks: int = 1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if scale != 1:
            raise ValueError(
                "MoSRv2Panels is a restoration/segmentation network and only "
                "supports scale=1."
            )
        if in_ch != 3 or out_ch != 3:
            raise ValueError(
                "MoSRv2Panels requires exactly 3 input and output channels."
            )
        if num_downsamples < 0:
            raise ValueError("num_downsamples must be non-negative.")
        expected_levels = num_downsamples + 1
        if len(encoder_dims) != expected_levels:
            raise ValueError(
                "encoder_dims must contain num_downsamples + 1 values: "
                f"expected {expected_levels}, got {len(encoder_dims)}."
            )
        if len(encoder_blocks) != expected_levels:
            raise ValueError(
                "encoder_blocks must contain num_downsamples + 1 values: "
                f"expected {expected_levels}, got {len(encoder_blocks)}."
            )
        if len(decoder_blocks) != expected_levels:
            raise ValueError(
                "decoder_blocks must contain num_downsamples + 1 values: "
                f"expected {expected_levels}, got {len(decoder_blocks)}."
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

        dims = tuple(encoder_dims)
        self.num_downsamples = num_downsamples
        self.pad_factor = 2**num_downsamples

        self.stem = nn.Conv2d(in_ch, dims[0], 3, padding=1)
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

    def _pad_input(self, x: Tensor) -> tuple[Tensor, int, int]:
        height, width = x.shape[-2:]
        pad_h = (self.pad_factor - height % self.pad_factor) % self.pad_factor
        pad_w = (self.pad_factor - width % self.pad_factor) % self.pad_factor
        if pad_h == 0 and pad_w == 0:
            return x, height, width
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), height, width

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("MoSRv2Panels expects input with shape [B, 3, H, W].")

        x, height, width = self._pad_input(x)
        input_rgb = x
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

        x = self.edge_refinement(x)
        residual = self.to_image(x)
        output = torch.cat(
            (
                input_rgb[:, 0:1] + residual[:, 0:1],
                residual[:, 1:2],
                input_rgb[:, 2:3] + residual[:, 2:3],
            ),
            dim=1,
        )
        return output[:, :, :height, :width]
