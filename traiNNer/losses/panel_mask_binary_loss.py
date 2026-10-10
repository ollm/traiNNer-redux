from typing import Any

from torch import Tensor, nn
from torch.nn import functional as F

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class PanelMaskBinaryLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # G channel: 0 = border, 1 (or 255) = panel.
        logits = pred[:, 1:2]
        target_mask = target[:, 1:2].to(dtype=logits.dtype)

        if target_mask.detach().amax() > 1:
            target_mask = target_mask / 255.0

        target_binary = (target_mask > 0.5).to(logits.dtype)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            target_binary,
        )

        return self.loss_weight * loss


def panel_mask_binary(
    loss_weight: float = 1.0,
    **kwargs: Any,
) -> PanelMaskBinaryLoss:
    return PanelMaskBinaryLoss(
        loss_weight=loss_weight,
        **kwargs,
    )


LOSS_REGISTRY.register(panel_mask_binary)