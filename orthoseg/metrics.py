"""Reproducible binary and multi-label 2-D segmentation metrics.

Dice and IoU are exact set scores (no smoothing). Both-empty masks score 1;
one-empty masks score 0. HD95 is the symmetric 95th percentile of distances
between *surface pixels*: both-empty masks score 0, while a one-empty pair
scores infinity. HD95 is in pixels by default. Pass pixel ``spacing`` in mm
to obtain HD95 in mm; do not label pixel distances as millimetres.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise TypeError("Masks must be numeric or boolean arrays")
    if not np.all(np.isfinite(array)):
        raise ValueError("Masks contain non-finite values")
    return array


def _binary_pair(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    pred = _as_numpy(prediction)
    true = _as_numpy(target)
    if pred.shape != true.shape:
        raise ValueError(f"Mask shapes differ: {pred.shape} and {true.shape}")
    if pred.ndim < 2:
        raise ValueError("Masks must have at least two spatial dimensions")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    return pred >= threshold, true >= 0.5


def binary_dice(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    *,
    threshold: float = 0.5,
) -> float:
    """Sørensen-Dice score; prediction is thresholded at ``threshold``."""
    pred, true = _binary_pair(prediction, target, threshold)
    denominator = int(pred.sum()) + int(true.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(pred, true).sum() / denominator)


def binary_iou(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    *,
    threshold: float = 0.5,
) -> float:
    """Intersection over union; prediction is thresholded at ``threshold``."""
    pred, true = _binary_pair(prediction, target, threshold)
    union = int(np.logical_or(pred, true).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, true).sum() / union)


def _spacing(spacing: float | Sequence[float] | None, ndim: int) -> tuple[float, ...]:
    if spacing is None:
        return (1.0,) * ndim
    if np.isscalar(spacing):
        values = (float(spacing),) * ndim
    else:
        values = tuple(float(value) for value in spacing)
    if len(values) != ndim or not all(np.isfinite(value) and value > 0 for value in values):
        raise ValueError(f"spacing must have {ndim} positive, finite values")
    return values


def hd95(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    *,
    threshold: float = 0.5,
    spacing: float | Sequence[float] | None = None,
) -> float:
    """Symmetric 95th-percentile Hausdorff distance between mask surfaces.

    ``spacing`` follows array-axis order: (row, column) in 2-D. Values must
    be mm per pixel if the returned physical distance is to be reported in mm.
    """
    pred, true = _binary_pair(prediction, target, threshold)
    if pred.ndim not in (2, 3):
        raise ValueError("hd95 expects one 2-D or 3-D mask, not a batch")
    sampling = _spacing(spacing, pred.ndim)
    pred_any = bool(pred.any())
    true_any = bool(true.any())
    if not pred_any and not true_any:
        return 0.0
    if pred_any != true_any:
        return float("inf")

    connectivity = generate_binary_structure(pred.ndim, 1)
    pred_surface = np.logical_xor(
        pred, binary_erosion(pred, structure=connectivity, border_value=0)
    )
    true_surface = np.logical_xor(
        true, binary_erosion(true, structure=connectivity, border_value=0)
    )
    distance_to_true = distance_transform_edt(~true_surface, sampling=sampling)
    distance_to_pred = distance_transform_edt(~pred_surface, sampling=sampling)
    distances = np.concatenate((
        distance_to_true[pred_surface], distance_to_pred[true_surface]
    ))
    return float(np.percentile(distances, 95))


def _as_nchw(value: np.ndarray | torch.Tensor) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim == 2:
        return array[None, None]
    if array.ndim == 3:
        return array[None]
    if array.ndim == 4:
        return array
    raise ValueError("Expected [H,W], [C,H,W], or [N,C,H,W] masks")


def _sample_spacings(
    spacing: float | Sequence[float] | np.ndarray | torch.Tensor | None,
    batch_size: int,
) -> list[tuple[float, float] | None]:
    if spacing is None:
        return [None] * batch_size
    if isinstance(spacing, torch.Tensor):
        spacing = spacing.detach().cpu().numpy()
    values = np.asarray(spacing, dtype=np.float64)
    if values.ndim == 0:
        parsed = _spacing(float(values), 2)
        return [parsed] * batch_size
    if values.ndim == 1:
        parsed = _spacing(values, 2)
        return [parsed] * batch_size
    if values.ndim == 2 and values.shape == (batch_size, 2):
        return [_spacing(row, 2) for row in values]
    raise ValueError("spacing must be scalar, [row,column], or [N,2]")


def evaluate_masks(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    *,
    threshold: float = 0.5,
    spacing: float | Sequence[float] | np.ndarray | torch.Tensor | None = None,
) -> dict[str, object]:
    """Compute per-class and macro-mean Dice, IoU, and HD95.

    Inputs are probabilities/binary masks with shape [H,W], [C,H,W], or
    [N,C,H,W]. Scores are first averaged over samples for each class, then
    averaged over classes. For fundus, use overlapping OD/OC channels; never
    use a softmax or make the cup mutually exclusive with the disc. ``spacing``
    may be one (row, column) pair for every sample or an [N,2] array of
    per-sample mm/pixel values, such as the dataset's resized_spacing field.
    """
    pred = _as_nchw(prediction)
    true = _as_nchw(target)
    if pred.shape != true.shape:
        raise ValueError(f"Mask shapes differ: {pred.shape} and {true.shape}")
    if pred.shape[0] == 0 or pred.shape[1] == 0:
        raise ValueError("Mask batch and channel dimensions must be nonempty")
    sample_spacings = _sample_spacings(spacing, pred.shape[0])

    per_class: dict[str, list[float]] = {"dice": [], "iou": [], "hd95": []}
    for class_index in range(pred.shape[1]):
        dice_values: list[float] = []
        iou_values: list[float] = []
        hd95_values: list[float] = []
        for batch_index in range(pred.shape[0]):
            prediction_mask = pred[batch_index, class_index]
            target_mask = true[batch_index, class_index]
            dice_values.append(binary_dice(prediction_mask, target_mask, threshold=threshold))
            iou_values.append(binary_iou(prediction_mask, target_mask, threshold=threshold))
            hd95_values.append(hd95(
                prediction_mask, target_mask, threshold=threshold,
                spacing=sample_spacings[batch_index]
            ))
        per_class["dice"].append(float(np.mean(dice_values)))
        per_class["iou"].append(float(np.mean(iou_values)))
        per_class["hd95"].append(float(np.mean(hd95_values)))
    mean = {name: float(np.mean(values)) for name, values in per_class.items()}
    return {"per_class": per_class, "mean": mean, "unit": "mm" if spacing is not None else "pixel"}
