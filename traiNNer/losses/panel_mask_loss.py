import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.losses.basic_loss import charbonnier_loss
from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class PanelMaskLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float,
        rgb_weight: float = 1.0,
        mask_bce_weight: float = 2.0,
        mask_dice_weight: float = 1.0,
        eps: float = 1e-6,
        charbonnier_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.rgb_weight = rgb_weight
        self.mask_bce_weight = mask_bce_weight
        self.mask_dice_weight = mask_dice_weight
        self.eps = eps
        self.charbonnier_eps = charbonnier_eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.ndim != 4 or target.ndim != 4:
            raise ValueError(
                "PanelMaskLoss expects pred and target with shape [B, C, H, W]."
            )
        if pred.shape != target.shape:
            raise ValueError("PanelMaskLoss expects pred and target to have the same shape.")
        if pred.shape[1] != 3:
            raise ValueError("PanelMaskLoss expects exactly 3 channels (RGB).")

        rgb_loss = charbonnier_loss(
            pred[:, (0, 2)],
            target[:, (0, 2)],
            eps=self.charbonnier_eps,
            reduction="mean",
        )

        mask_logits = pred[:, 1:2]
        mask_target = target[:, 1:2]
        target_scale = torch.where(
            mask_target.detach().amax() > 1,
            mask_target.new_tensor(255.0),
            mask_target.new_tensor(1.0),
        )
        mask_target = mask_target / target_scale

        bce_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_target)

        mask_probabilities = torch.sigmoid(mask_logits)
        pred_flat = mask_probabilities.flatten(1)
        target_flat = mask_target.flatten(1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2 * intersection + self.eps) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.eps
        )
        dice_loss = 1 - dice.mean()

        return (
            self.rgb_weight * rgb_loss
            + self.mask_bce_weight * bce_loss
            + self.mask_dice_weight * dice_loss
        )
