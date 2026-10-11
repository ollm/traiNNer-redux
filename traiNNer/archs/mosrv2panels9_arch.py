import torch
from torch import Tensor

from traiNNer.archs.mosrv2panels8_arch import MoSRv2Panels8
from traiNNer.utils.panel_mask_util import reconstruct_panel_regions
from traiNNer.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class MoSRv2Panels9(MoSRv2Panels8):
    """Panels8 logits with perimeter-guided region reconstruction at inference.

    The trained forward path and parameter names match Panels8. Reconstruction
    fills only uncertain pixels enclosed by confident boundaries; it does not
    invent missing boundaries for large gaps or unframed panels.
    """

    @torch.no_grad()
    def post_process_inference(self, output: Tensor) -> Tensor:
        if output.ndim != 4 or output.shape[1] != 3:
            raise ValueError("MoSRv2Panels9 expects output with shape [B, 3, H, W].")

        panel_mask = reconstruct_panel_regions(output[:, 1:2])
        return torch.cat((output[:, 0:1], panel_mask, output[:, 2:3]), dim=1)


__all__ = ["MoSRv2Panels9"]
