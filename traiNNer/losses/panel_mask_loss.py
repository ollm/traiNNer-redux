from typing import Any

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
        mask_gradient_weight: float = 1.0,
        interior_smoothness_weight: float = 0.5,
        border_continuity_weight: float = 1.0,
        mask_multiscale_weight: float = 0.25,
        mask_curvature_weight: float = 0.25,
        mask_positive_weight: float = 2.0,
        eps: float = 1e-6,
        charbonnier_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.rgb_weight = rgb_weight
        self.mask_bce_weight = mask_bce_weight
        self.mask_dice_weight = mask_dice_weight
        self.mask_gradient_weight = mask_gradient_weight
        self.interior_smoothness_weight = interior_smoothness_weight
        self.border_continuity_weight = border_continuity_weight
        self.mask_multiscale_weight = mask_multiscale_weight
        self.mask_curvature_weight = mask_curvature_weight
        self.mask_positive_weight = mask_positive_weight
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

        positive_fraction = mask_target.mean().detach()
        class_balance = ((1 - positive_fraction) / (positive_fraction + self.eps)).clamp(
            min=1.0, max=self.mask_positive_weight
        )
        bce = F.binary_cross_entropy_with_logits(
            mask_logits, mask_target, reduction="none"
        )
        bce_weights = torch.where(mask_target > 0.5, class_balance, 1.0)
        bce_loss = (bce * bce_weights).mean()

        neighbor_kernel = mask_target.new_ones((1, 1, 3, 3))
        neighbor_kernel[:, :, 1, 1] = 0
        target_neighbors = F.conv2d(mask_target, neighbor_kernel, padding=1)
        continuity_mask = mask_target * (target_neighbors > 0).to(mask_target.dtype)
        continuity_loss = (bce * continuity_mask).sum() / (
            continuity_mask.sum() + self.eps
        )

        mask_probabilities = torch.sigmoid(mask_logits)
        pred_flat = mask_probabilities.flatten(1)
        target_flat = mask_target.flatten(1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2 * intersection + self.eps) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.eps
        )
        dice_loss = 1 - dice.mean()

        pred_dx = mask_probabilities[:, :, :, 1:] - mask_probabilities[:, :, :, :-1]
        target_dx = mask_target[:, :, :, 1:] - mask_target[:, :, :, :-1]
        pred_dy = mask_probabilities[:, :, 1:, :] - mask_probabilities[:, :, :-1, :]
        target_dy = mask_target[:, :, 1:, :] - mask_target[:, :, :-1, :]
        gradient_loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

        panel_x_weight = (1 - mask_target[:, :, :, 1:]).clamp_min(0)
        panel_x_weight = panel_x_weight * (1 - mask_target[:, :, :, :-1]).clamp_min(0)
        panel_y_weight = (1 - mask_target[:, :, 1:, :]).clamp_min(0)
        panel_y_weight = panel_y_weight * (1 - mask_target[:, :, :-1, :]).clamp_min(0)
        interior_smoothness = (
            (torch.abs(pred_dx) * panel_x_weight).sum()
            + (torch.abs(pred_dy) * panel_y_weight).sum()
        ) / (panel_x_weight.sum() + panel_y_weight.sum() + self.eps)

        if min(mask_probabilities.shape[-2:]) >= 3:
            pred_dxx = (
                mask_probabilities[:, :, :, 2:]
                - 2 * mask_probabilities[:, :, :, 1:-1]
                + mask_probabilities[:, :, :, :-2]
            )
            target_dxx = (
                mask_target[:, :, :, 2:]
                - 2 * mask_target[:, :, :, 1:-1]
                + mask_target[:, :, :, :-2]
            )
            pred_dyy = (
                mask_probabilities[:, :, 2:, :]
                - 2 * mask_probabilities[:, :, 1:-1, :]
                + mask_probabilities[:, :, :-2, :]
            )
            target_dyy = (
                mask_target[:, :, 2:, :]
                - 2 * mask_target[:, :, 1:-1, :]
                + mask_target[:, :, :-2, :]
            )
            curvature_loss = F.l1_loss(pred_dxx, target_dxx) + F.l1_loss(
                pred_dyy, target_dyy
            )
        else:
            curvature_loss = mask_probabilities.new_zeros(())

        multiscale_loss = mask_probabilities.new_zeros(())
        for scale in (2, 4):
            if min(mask_probabilities.shape[-2:]) >= scale:
                coarse_prediction = F.avg_pool2d(
                    mask_probabilities, kernel_size=scale, stride=scale
                )
                coarse_target = F.avg_pool2d(mask_target, kernel_size=scale, stride=scale)
                multiscale_loss = multiscale_loss + F.l1_loss(
                    coarse_prediction, coarse_target
                )

        return (
            self.rgb_weight * rgb_loss
            + self.mask_bce_weight * bce_loss
            + self.mask_dice_weight * dice_loss
            + self.mask_gradient_weight * gradient_loss
            + self.interior_smoothness_weight * interior_smoothness
            + self.border_continuity_weight * continuity_loss
            + self.mask_multiscale_weight * multiscale_loss
            + self.mask_curvature_weight * curvature_loss
        )


def panel_mask(loss_weight: float, **kwargs: Any) -> PanelMaskLoss:
    return PanelMaskLoss(loss_weight=loss_weight, **kwargs)


LOSS_REGISTRY.register(panel_mask)
