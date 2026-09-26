"""Training and evaluation for the paper-guided OrthoSeg implementation.

The paper does not define every encoder, decoder, interpolation, or optimizer
detail. This module makes those choices explicit so experiments can be audited.
It does not contain reported paper measurements or bundled dataset assets.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ManifestDataset, validate_disjoint_manifests
from .indexed_data import prepare_indexed_manifest
from .losses import (
    ConditionalGaussianMI,
    mid_weight,
    segmentation_loss,
    structure_equivariance_loss,
    texture_invariance_loss,
)
from .metrics import evaluate_masks
from .model import OrthoSeg


def _resolve_path(base: Path, value: str | os.PathLike[str]) -> Path:
    raw = str(value)
    if os.name != "nt" and PureWindowsPath(raw).drive:
        raise ValueError(f"A Windows absolute path cannot be used on this host: {raw}")
    path = Path(raw.replace("\\", "/")).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a JSON configuration; all relative paths use its parent directory."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a JSON object")
    for section in ("data", "model", "train"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration needs a {section!r} object")
    config["_config_path"] = str(config_path)
    return config


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"{name!r} must be an object")
    return value


def _data_paths(config: Mapping[str, Any]) -> tuple[Path, dict[str, Path]]:
    """Resolve explicit manifests or prepare indexed CSV sidecars once per call."""
    base = Path(str(config["_config_path"])).parent
    data = _section(config, "data")
    indexed_datasets = data.get("indexed_datasets")
    if indexed_datasets is not None:
        if any(key in data for key in ("root", "train_manifest", "val_manifest", "test_manifest")):
            raise ValueError(
                "data.indexed_datasets cannot be combined with data.root or explicit manifests"
            )
        if not isinstance(indexed_datasets, Mapping) or not indexed_datasets:
            raise ValueError("data.indexed_datasets must be a nonempty split-to-dataset object")
        invalid_splits = set(indexed_datasets) - {"train", "val", "test"}
        if invalid_splits:
            raise ValueError(f"Unknown indexed split names: {sorted(invalid_splits)}")
        for key in ("indexed_data_root", "indexed_metadata_root"):
            value = data.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"data.{key} must be a nonempty path in indexed mode")
        root = _resolve_path(base, data["indexed_data_root"])
        metadata_root = _resolve_path(base, data["indexed_metadata_root"])
        sidecar_dir = _output_dir(config) / "indexed_manifests"
        manifests: dict[str, Path] = {}
        for split in ("train", "val", "test"):
            if split not in indexed_datasets:
                continue
            dataset = indexed_datasets[split]
            if not isinstance(dataset, str) or not dataset.strip():
                raise ValueError(f"data.indexed_datasets.{split} must be a nonempty name")
            manifests[split] = prepare_indexed_manifest(
                dataset.strip(), split,
                data_root=root,
                metadata_root=metadata_root,
                output_dir=sidecar_dir,
            )
        return root, manifests

    if "indexed_data_root" in data or "indexed_metadata_root" in data:
        raise ValueError("Indexed roots require data.indexed_datasets")
    root = _resolve_path(base, str(data.get("root", "../data")))
    manifests: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        value = data.get(f"{split}_manifest")
        if value:
            manifests[split] = _resolve_path(base, str(value))
    return root, manifests


def _image_size(data: Mapping[str, Any]) -> tuple[int, int]:
    value = data.get("image_size", 352)
    if isinstance(value, int):
        size = (value, value)
    elif isinstance(value, list) and len(value) == 2:
        size = tuple(value)
    else:
        raise ValueError("data.image_size must be an integer or [height, width]")
    if any(not isinstance(side, int) or side <= 0 for side in size):
        raise ValueError("data.image_size dimensions must be positive integers")
    return size


def _task(config: Mapping[str, Any]) -> tuple[str, int, str]:
    data = _section(config, "data")
    model = _section(config, "model")
    task = str(data.get("task", "binary"))
    if task not in ("binary", "fundus"):
        raise ValueError("data.task must be 'binary' or 'fundus'")
    classes = 1 if task == "binary" else 2
    if int(model.get("num_classes", classes)) != classes:
        raise ValueError(f"model.num_classes must be {classes} for task {task!r}")
    if int(model.get("in_channels", 3)) != 3:
        raise ValueError("The manifest loader converts images to RGB; model.in_channels must be 3")
    return task, classes, "binary" if classes == 1 else "multilabel"


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, CUDA, and DataLoader workers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _loader(
    config: Mapping[str, Any], split: str, *, device: torch.device, seed: int,
    prepared_paths: tuple[Path, dict[str, Path]] | None = None,
) -> DataLoader:
    data = _section(config, "data")
    train = _section(config, "train")
    task, classes, _ = _task(config)
    root, manifests = prepared_paths if prepared_paths is not None else _data_paths(config)
    if split not in manifests:
        raise ValueError(f"data.{split}_manifest is required")
    dataset = ManifestDataset(
        manifests[split],
        root=root,
        split=split,
        num_classes=classes,
        size=_image_size(data),
        augment=split == "train",
    )
    if task == "fundus" and classes != 2:
        raise AssertionError("Fundus masks must have disc and cup channels")
    batch_size = int(train.get("batch_size", 16)) if split == "train" else 1
    workers = int(train.get("num_workers", 0))
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if split == "train" and len(dataset) < 2:
        raise ValueError("MID training needs at least two training images")
    # Eq. 2 needs an off-diagonal pair; avoid a last batch with one image.
    drop_last = split == "train" and len(dataset) > batch_size and len(dataset) % batch_size == 1
    if split == "train" and batch_size < 2:
        raise ValueError("MID training needs train.batch_size >= 2")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
        generator=generator,
    )


@dataclass(frozen=True)
class AffineView:
    """One sampled pixel-space geometry shared by the image and feature scales."""

    cosine: Tensor
    sine: Tensor
    horizontal: Tensor
    vertical: Tensor

    def _grid(self, tensor: Tensor) -> Tensor:
        if tensor.ndim != 4 or tensor.shape[0] != self.cosine.shape[0]:
            raise ValueError("AffineView expects a matching [N,C,H,W] tensor")
        batch, _, height, width = tensor.shape
        cosine = self.cosine.to(device=tensor.device, dtype=tensor.dtype)
        sine = self.sine.to(device=tensor.device, dtype=tensor.dtype)
        horizontal = self.horizontal.to(device=tensor.device, dtype=tensor.dtype)
        vertical = self.vertical.to(device=tensor.device, dtype=tensor.dtype)
        # affine_grid uses normalized x/y coordinates. Rebuild the cross terms
        # for each feature resolution so rectangular and odd-sized maps retain
        # the same rotation and scale in pixel coordinates.
        theta = torch.zeros(batch, 2, 3, device=tensor.device, dtype=tensor.dtype)
        theta[:, 0, 0] = cosine * horizontal
        theta[:, 0, 1] = -sine * vertical * (height / width)
        theta[:, 1, 0] = sine * horizontal * (width / height)
        theta[:, 1, 1] = cosine * vertical
        return F.affine_grid(theta, tensor.shape, align_corners=False)

    def __call__(self, tensor: Tensor) -> Tensor:
        grid = self._grid(tensor)
        return F.grid_sample(tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def valid_mask(self, tensor: Tensor) -> Tensor:
        """Identify pixels fully supported by the source, excluding zero padding."""
        grid = self._grid(tensor)
        coverage = F.grid_sample(
            torch.ones_like(tensor[:, :1]), grid, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )
        return coverage >= 0.9999


def sample_affine_view(images: Tensor) -> AffineView:
    """Sample per-image ±15° rotation, 0.8–1.2 scaling, and two flips."""
    batch = images.shape[0]
    device = images.device
    angle = (torch.rand(batch, device=device) * 30.0 - 15.0) * (math.pi / 180.0)
    scale = torch.rand(batch, device=device) * 0.4 + 0.8
    horizontal = torch.where(torch.rand(batch, device=device) < 0.5, -1.0, 1.0)
    vertical = torch.where(torch.rand(batch, device=device) < 0.5, -1.0, 1.0)
    cosine, sine = angle.cos() / scale, angle.sin() / scale
    return AffineView(cosine, sine, horizontal, vertical)


def build_model(config: Mapping[str, Any], device: torch.device) -> tuple[OrthoSeg, ConditionalGaussianMI]:
    _, classes, _ = _task(config)
    model_config = _section(config, "model")
    widths = tuple(int(value) for value in model_config.get("widths", (16, 32, 64, 128)))
    latent_dim = int(model_config.get("latent_dim", 256))
    model = OrthoSeg(
        in_channels=int(model_config.get("in_channels", 3)),
        num_classes=classes,
        widths=widths,
        latent_dim=latent_dim,
        num_experts=int(model_config.get("num_experts", 4)),
    ).to(device)
    estimator = ConditionalGaussianMI(latent_dim=latent_dim).to(device)
    return model, estimator


def train_one_epoch(
    model: OrthoSeg,
    estimator: ConditionalGaussianMI,
    loader: DataLoader,
    model_optimizer: torch.optim.Optimizer,
    auxiliary_optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epoch: int,
    epochs: int,
    task: str,
    alpha: float,
    beta: float,
    tau: float,
    gamma: float,
) -> dict[str, float]:
    """One epoch with an auxiliary q_phi step followed by a model step."""
    model.train()
    estimator.train()
    totals = {name: 0.0 for name in ("total", "seg", "mid", "se", "ti", "aux")}
    seen = 0
    weight = mid_weight(epoch, total_epochs=epochs, tau=tau, gamma=gamma)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if images.shape[0] < 2:
            raise RuntimeError("MID encountered a singleton batch; use batch_size >= 2")
        view = sample_affine_view(images)

        model_optimizer.zero_grad(set_to_none=True)
        original = model(images, return_features=True)
        auxiliary_optimizer.zero_grad(set_to_none=True)
        auxiliary_loss = estimator.auxiliary_loss(
            original.structure_latent, original.texture_latent
        )
        auxiliary_loss.backward()
        auxiliary_optimizer.step()

        transformed = model(view(images), return_features=True)
        seg = segmentation_loss(original.logits, masks, task=task)
        mid = estimator.mid_loss(original.structure_latent, original.texture_latent)
        se = structure_equivariance_loss(
            original.structure_features, transformed.structure_features, view
        )
        ti = texture_invariance_loss(original.texture_latent, transformed.texture_latent)
        total = seg + weight * mid + alpha * se + beta * ti
        total.backward()
        model_optimizer.step()
        auxiliary_optimizer.zero_grad(set_to_none=True)

        size = int(images.shape[0])
        for name, value in (("total", total), ("seg", seg), ("mid", mid),
                            ("se", se), ("ti", ti), ("aux", auxiliary_loss)):
            totals[name] += float(value.detach().item()) * size
        seen += size
    if seen == 0:
        raise RuntimeError("Training loader yielded no samples")
    return {name: value / seen for name, value in totals.items()} | {"mid_weight": weight}


def _sample_spacing(batch: Mapping[str, Any], index: int, constant_spacing: Any) -> list[float] | None:
    if "resized_spacing" in batch:
        if constant_spacing is not None:
            raise ValueError("Specify spacing in each manifest or data.spacing, not both")
        return [float(value) for value in batch["resized_spacing"][index].tolist()]
    if constant_spacing is None:
        return None
    if not isinstance(constant_spacing, list) or len(constant_spacing) != 2:
        raise ValueError("data.spacing must be null or [row_mm_per_output_pixel, column_mm_per_output_pixel]")
    return [float(value) for value in constant_spacing]


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    constant_spacing: Any = None,
) -> dict[str, Any]:
    """Return per-image scores and class/sample macro averages."""
    model.eval()
    collected = {name: [[] for _ in range(num_classes)] for name in ("dice", "iou", "hd95")}
    cases: list[dict[str, Any]] = []
    unit: str | None = None
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"]
            probabilities = torch.sigmoid(model(images)).cpu()
            for index in range(images.shape[0]):
                spacing = _sample_spacing(batch, index, constant_spacing)
                result = evaluate_masks(probabilities[index], masks[index], spacing=spacing)
                if unit is not None and result["unit"] != unit:
                    raise ValueError("Evaluation samples mix physical and pixel HD95 units")
                unit = str(result["unit"])
                for metric in collected:
                    for class_index, value in enumerate(result["per_class"][metric]):
                        collected[metric][class_index].append(float(value))
                cases.append({
                    "image_path": batch["image_path"][index],
                    "per_class": result["per_class"],
                    "mean": result["mean"],
                })
    if not cases:
        raise RuntimeError("Evaluation loader yielded no samples")
    per_class = {
        name: [sum(scores) / len(scores) for scores in class_scores]
        for name, class_scores in collected.items()
    }
    mean = {name: sum(values) / len(values) for name, values in per_class.items()}
    return {"samples": len(cases), "per_class": per_class, "mean": mean, "unit": unit, "cases": cases}


def _json_compatible(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else ("-Infinity" if value < 0 else "NaN")
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_compatible(dict(payload)), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid OrthoSeg checkpoint: {path}")
    return checkpoint


def _output_dir(config: Mapping[str, Any]) -> Path:
    base = Path(str(config["_config_path"])).parent
    train = _section(config, "train")
    return _resolve_path(base, str(train.get("output_dir", "../runs/example")))


def _image_key(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").casefold()


def train_experiment(config: Mapping[str, Any]) -> Path:
    """Train one source domain and select the highest validation Dice checkpoint."""
    train_config = _section(config, "train")
    data_config = _section(config, "data")
    _, classes, loss_task = _task(config)
    indexed_sources = data_config.get("indexed_datasets")
    if isinstance(indexed_sources, Mapping) and {"train", "val"}.issubset(indexed_sources):
        if str(indexed_sources["train"]).strip() != str(indexed_sources["val"]).strip():
            raise ValueError("Indexed training and validation must use the same source dataset")
    epochs = int(train_config.get("epochs", 200))
    if epochs < 1:
        raise ValueError("train.epochs must be positive")
    for name in ("alpha", "beta", "tau"):
        if float(train_config.get(name, {"alpha": 0.3, "beta": 1.0, "tau": 1.0}[name])) < 0:
            raise ValueError(f"train.{name} must be nonnegative")
    seed = int(train_config.get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root, manifests = _data_paths(config)
    if "train" not in manifests or "val" not in manifests:
        raise ValueError("Training requires train_manifest and val_manifest")
    validate_disjoint_manifests(manifests, root=root)
    prepared_paths = (root, manifests)
    train_loader = _loader(config, "train", device=device, seed=seed, prepared_paths=prepared_paths)
    val_loader = _loader(config, "val", device=device, seed=seed, prepared_paths=prepared_paths)
    model, estimator = build_model(config, device)
    lr = float(train_config.get("lr", 1e-4))
    aux_lr = float(train_config.get("aux_lr", lr))
    weight_decay = float(train_config.get("weight_decay", 1e-4))
    if min(lr, aux_lr) <= 0 or weight_decay < 0:
        raise ValueError("learning rates must be positive and weight decay nonnegative")
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    auxiliary_optimizer = torch.optim.AdamW(estimator.parameters(), lr=aux_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=epochs)
    auxiliary_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(auxiliary_optimizer, T_max=epochs)
    output_dir = _output_dir(config)
    existing = [name for name in ("history.jsonl", "last.pt", "best.pt") if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            f"Training output already exists in {output_dir}: {', '.join(existing)}; "
            "choose a new train.output_dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_image_paths = sorted({_image_key(record.image_path) for record in train_loader.dataset.records})
    val_image_paths = sorted({_image_key(record.image_path) for record in val_loader.dataset.records})
    best_dice = float("-inf")
    history_path = output_dir / "history.jsonl"
    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(epochs):
            current_lr = model_optimizer.param_groups[0]["lr"]
            losses = train_one_epoch(
                model, estimator, train_loader, model_optimizer, auxiliary_optimizer,
                device=device, epoch=epoch, epochs=epochs, task=loss_task,
                alpha=float(train_config.get("alpha", 0.3)),
                beta=float(train_config.get("beta", 1.0)),
                tau=float(train_config.get("tau", 1.0)),
                gamma=float(train_config.get("gamma", 0.9)),
            )
            validation = evaluate_model(
                model, val_loader, device=device, num_classes=classes,
                constant_spacing=data_config.get("spacing"),
            )
            scheduler.step()
            auxiliary_scheduler.step()
            val_dice = float(validation["mean"]["dice"])
            improved = val_dice > best_dice
            if improved:
                best_dice = val_dice
            record = {
                "epoch": epoch + 1, "train": losses,
                "val": {key: value for key, value in validation.items() if key != "cases"},
                "lr": current_lr,
            }
            history.write(json.dumps(_json_compatible(record), ensure_ascii=False, allow_nan=False) + "\n")
            history.flush()
            checkpoint = {
                "epoch": epoch,
                "best_val_dice": best_dice,
                "train_image_paths": train_image_paths,
                "val_image_paths": val_image_paths,
                "config": {key: value for key, value in config.items() if not key.startswith("_")},
                "model_state_dict": model.state_dict(),
                "estimator_state_dict": estimator.state_dict(),
                "model_optimizer_state_dict": model_optimizer.state_dict(),
                "auxiliary_optimizer_state_dict": auxiliary_optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "auxiliary_scheduler_state_dict": auxiliary_scheduler.state_dict(),
            }
            _save_checkpoint(output_dir / "last.pt", checkpoint)
            if improved:
                _save_checkpoint(output_dir / "best.pt", checkpoint)
    return output_dir / "best.pt"


def evaluate_experiment(
    config: Mapping[str, Any], *, checkpoint_path: Path, split: str = "test"
) -> Path:
    if split not in ("val", "test"):
        raise ValueError("split must be val or test")
    seed = int(_section(config, "train").get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, classes, _ = _task(config)
    root, manifests = _data_paths(config)
    if split not in manifests:
        raise ValueError(f"data.{split}_manifest is required")
    checkpoint = _load_checkpoint(checkpoint_path, device)
    # Check current manifests and the source paths recorded when the checkpoint
    # was trained. This catches path-identical leakage even with a target-only
    # evaluation config; content-identical images at new paths require a data audit.
    validate_disjoint_manifests(manifests, root=root)
    loader = _loader(
        config, split, device=device, seed=seed, prepared_paths=(root, manifests)
    )
    train_paths = checkpoint.get("train_image_paths")
    val_paths = checkpoint.get("val_image_paths")
    if not isinstance(train_paths, list) or not isinstance(val_paths, list):
        raise ValueError("Checkpoint lacks source image identifiers for leakage checking")
    source_set = set(train_paths)
    if split == "test":
        source_set.update(val_paths)
    for record in loader.dataset.records:
        if _image_key(record.image_path) in source_set:
            raise ValueError(f"Evaluation image overlaps checkpoint source data: {record.image_path}")
    model, _ = build_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    results = evaluate_model(
        model, loader, device=device, num_classes=classes,
        constant_spacing=_section(config, "data").get("spacing"),
    )
    output_path = _output_dir(config) / f"evaluation_{split}.json"
    write_json(output_path, results)
    return output_path
