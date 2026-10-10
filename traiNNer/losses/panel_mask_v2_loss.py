
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class PanelMaskLossV2(nn.Module):
    def __init__(
        self,
        loss_weight: float = 1.0,
        mask_bce_weight: float = 2.0,
        mask_dice_weight: float = 1.0,
        mask_gradient_weight: float = 1.0,
        interior_smoothness_weight: float = 0.25,
        mask_multiscale_weight: float = 0.25,
        max_class_weight: float = 3.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.loss_weight = loss_weight
        self.mask_bce_weight = mask_bce_weight
        self.mask_dice_weight = mask_dice_weight
        self.mask_gradient_weight = mask_gradient_weight
        self.interior_smoothness_weight = interior_smoothness_weight
        self.mask_multiscale_weight = mask_multiscale_weight
        self.max_class_weight = max_class_weight
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.ndim != 4 or target.ndim != 4:
            raise ValueError(
                'pred and target must have shape (N, C, H, W)'
            )

        if pred.shape != target.shape:
            raise ValueError(
                f'pred and target shapes must match: '
                f'{pred.shape} != {target.shape}'
            )

        if pred.shape[1] != 3:
            raise ValueError(
                f'Expected 3 channels, got {pred.shape[1]}'
            )

        # Channel G contains the mask:
        # 0 = border, 1 (or 255) = panel.
        mask_logits = pred[:, 1:2]
        mask_target = target[:, 1:2].to(dtype=pred.dtype)

        # Support targets encoded as either [0, 1] or [0, 255].
        target_max = mask_target.detach().amax(
            dim=(1, 2, 3),
            keepdim=True,
        )

        target_scale = torch.where(
            target_max > 1,
            torch.full_like(target_max, 255.0),
            torch.ones_like(target_max),
        )

        mask_target = (mask_target / target_scale).clamp(0, 1)

        # ---------------------------------------------------------
        # 1. Class-balanced binary cross-entropy.
        # ---------------------------------------------------------
        # Balance both classes rather than only increasing the
        # weight of positive pixels.
        positive_fraction = mask_target.detach().mean(
            dim=(1, 2, 3),
            keepdim=True,
        ).clamp(self.eps, 1.0 - self.eps)

        negative_fraction = 1.0 - positive_fraction

        positive_weight = (
            0.5 / positive_fraction
        ).clamp(max=self.max_class_weight)

        negative_weight = (
            0.5 / negative_fraction
        ).clamp(max=self.max_class_weight)

        class_weights = torch.where(
            mask_target > 0.5,
            positive_weight,
            negative_weight,
        )

        bce = F.binary_cross_entropy_with_logits(
            mask_logits,
            mask_target,
            reduction='none',
        )

        bce_loss = (
            (bce * class_weights).sum()
            / (class_weights.sum() + self.eps)
        )

        # ---------------------------------------------------------
        # 2. Dice loss.
        # ---------------------------------------------------------
        mask_probabilities = torch.sigmoid(mask_logits)

        pred_flat = mask_probabilities.flatten(1)
        target_flat = mask_target.flatten(1)

        intersection = (pred_flat * target_flat).sum(dim=1)

        dice_score = (
            2.0 * intersection + self.eps
        ) / (
            pred_flat.sum(dim=1)
            + target_flat.sum(dim=1)
            + self.eps
        )

        dice_loss = 1.0 - dice_score.mean()

        # ---------------------------------------------------------
        # 3. Gradient loss.
        # ---------------------------------------------------------
        # Match the predicted mask transitions to the target
        # transitions, helping the network learn sharper borders.
        gradient_loss = mask_probabilities.new_zeros(())

        if mask_probabilities.shape[-1] > 1:
            pred_dx = (
                mask_probabilities[:, :, :, 1:]
                - mask_probabilities[:, :, :, :-1]
            )

            target_dx = (
                mask_target[:, :, :, 1:]
                - mask_target[:, :, :, :-1]
            )

            gradient_loss = gradient_loss + F.l1_loss(
                pred_dx,
                target_dx,
            )

        if mask_probabilities.shape[-2] > 1:
            pred_dy = (
                mask_probabilities[:, :, 1:, :]
                - mask_probabilities[:, :, :-1, :]
            )

            target_dy = (
                mask_target[:, :, 1:, :]
                - mask_target[:, :, :-1, :]
            )

            gradient_loss = gradient_loss + F.l1_loss(
                pred_dy,
                target_dy,
            )

        # ---------------------------------------------------------
        # 4. Smoothness inside panel interiors.
        # ---------------------------------------------------------
        # Only penalize changes between neighboring pixels that
        # are both labeled as panel pixels in the target.
        # Border pixels (target = 0) are not treated as interiors.
        interior_smoothness = mask_probabilities.new_zeros(())
        smoothness_sum = mask_probabilities.new_zeros(())
        smoothness_count = mask_probabilities.new_zeros(())

        if mask_probabilities.shape[-1] > 1:
            pred_dx = (
                mask_probabilities[:, :, :, 1:]
                - mask_probabilities[:, :, :, :-1]
            )

            panel_x_weight = (
                (mask_target[:, :, :, 1:] > 0.5)
                & (mask_target[:, :, :, :-1] > 0.5)
            ).to(dtype=pred.dtype)

            smoothness_sum = smoothness_sum + (
                pred_dx.abs() * panel_x_weight
            ).sum()

            smoothness_count = smoothness_count + panel_x_weight.sum()

        if mask_probabilities.shape[-2] > 1:
            pred_dy = (
                mask_probabilities[:, :, 1:, :]
                - mask_probabilities[:, :, :-1, :]
            )

            panel_y_weight = (
                (mask_target[:, :, 1:, :] > 0.5)
                & (mask_target[:, :, :-1, :] > 0.5)
            ).to(dtype=pred.dtype)

            smoothness_sum = smoothness_sum + (
                pred_dy.abs() * panel_y_weight
            ).sum()

            smoothness_count = smoothness_count + panel_y_weight.sum()

        interior_smoothness = (
            smoothness_sum / (smoothness_count + self.eps)
        )

        # ---------------------------------------------------------
        # 5. Multiscale loss.
        # ---------------------------------------------------------
        # Preserve the general mask shape at lower resolutions.
        multiscale_loss = mask_probabilities.new_zeros(())
        multiscale_count = 0

        for scale in (2, 4):
            if min(mask_probabilities.shape[-2:]) >= scale:
                coarse_prediction = F.avg_pool2d(
                    mask_probabilities,
                    kernel_size=scale,
                    stride=scale,
                )

                coarse_target = F.avg_pool2d(
                    mask_target,
                    kernel_size=scale,
                    stride=scale,
                )

                multiscale_loss = multiscale_loss + F.l1_loss(
                    coarse_prediction,
                    coarse_target,
                )

                multiscale_count += 1

        if multiscale_count:
            multiscale_loss = multiscale_loss / multiscale_count

        # ---------------------------------------------------------
        # 6. Total loss.
        # ---------------------------------------------------------
        total_loss = (
            self.mask_bce_weight * bce_loss
            + self.mask_dice_weight * dice_loss
            + self.mask_gradient_weight * gradient_loss
            + self.interior_smoothness_weight * interior_smoothness
            + self.mask_multiscale_weight * multiscale_loss
        )

        return self.loss_weight * total_loss


def panel_mask_v2(
    loss_weight: float = 1.0,
    **kwargs: Any,
) -> PanelMaskLossV2:
    return PanelMaskLossV2(
        loss_weight=loss_weight,
        **kwargs,
    )


LOSS_REGISTRY.register(panel_mask_v2)