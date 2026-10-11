import numpy as np
import torch
from numpy.typing import NDArray
from scipy import ndimage
from torch import Tensor
from torch.nn import functional as F  # noqa: N812


def panel_boundary_probability(mask_probabilities: Tensor) -> Tensor:
    if mask_probabilities.ndim != 4 or mask_probabilities.shape[1] != 1:
        raise ValueError("mask_probabilities must have shape [B, 1, H, W].")

    local_maximum = F.max_pool2d(mask_probabilities, kernel_size=3, stride=1, padding=1)
    local_minimum = -F.max_pool2d(
        -mask_probabilities, kernel_size=3, stride=1, padding=1
    )
    return local_maximum - local_minimum


@torch.no_grad()
def reconstruct_panel_regions(
    mask_logits: Tensor,
    panel_threshold: float = 0.5,
    repair_threshold: float = 0.25,
    boundary_threshold: float = 0.75,
) -> Tensor:
    """Fill uncertain pixels inside closed panels while preserving perimeter exterior."""
    if mask_logits.ndim != 4 or mask_logits.shape[1] != 1:
        raise ValueError("mask_logits must have shape [B, 1, H, W].")
    if mask_logits.shape[0] == 0 or min(mask_logits.shape[-2:]) == 0:
        raise ValueError(
            "mask_logits must have non-empty batch and spatial dimensions."
        )
    if not 0 <= repair_threshold <= panel_threshold <= 1:
        raise ValueError(
            "Thresholds must satisfy 0 <= repair_threshold <= panel_threshold <= 1."
        )
    if not 0 <= boundary_threshold <= 1:
        raise ValueError("boundary_threshold must be in [0, 1].")

    probabilities = torch.sigmoid(mask_logits.float())
    boundary_probabilities = panel_boundary_probability(probabilities)
    probability_arrays: NDArray[np.float32] = np.asarray(
        probabilities[:, 0].cpu().numpy(), dtype=np.float32
    )
    boundary_arrays: NDArray[np.float32] = np.asarray(
        boundary_probabilities[:, 0].cpu().numpy(), dtype=np.float32
    )
    connectivity: NDArray[np.bool_] = np.array(
        [[False, True, False], [True, True, True], [False, True, False]],
        dtype=bool,
    )
    closing_structure: NDArray[np.bool_] = np.ones((3, 3), dtype=bool)
    reconstructed_masks: list[np.ndarray] = []

    for probability, boundary_probability in zip(
        probability_arrays, boundary_arrays, strict=True
    ):
        boundary = boundary_probability >= boundary_threshold
        closed_boundary = ndimage.binary_closing(boundary, structure=closing_structure)
        exterior_candidates: NDArray[np.bool_] = np.logical_and(
            probability < panel_threshold, np.logical_not(closed_boundary)
        )
        labels: NDArray[np.int32] = np.empty(exterior_candidates.shape, dtype=np.int32)
        ndimage.label(exterior_candidates, structure=connectivity, output=labels)
        perimeter_labels = np.unique(
            np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
        )
        perimeter_labels = perimeter_labels[perimeter_labels != 0]
        exterior: NDArray[np.bool_] = np.isin(labels, perimeter_labels)

        panel = np.logical_or(
            probability >= panel_threshold,
            np.logical_and(
                np.logical_and(
                    probability >= repair_threshold, np.logical_not(exterior)
                ),
                np.logical_not(closed_boundary),
            ),
        )
        reconstructed_masks.append(panel.astype(np.float32))

    mask = torch.from_numpy(np.stack(reconstructed_masks)[:, None])
    return mask.to(device=mask_logits.device, dtype=mask_logits.dtype)
