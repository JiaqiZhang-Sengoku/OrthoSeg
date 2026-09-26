"""CSV based 2-D segmentation data loading.

The public manifest format uses paths relative to ``root`` (or absolute paths):

* Binary: ``image_path,mask_path``.
* Optic disc / cup: ``image_path,od_mask_path,oc_mask_path``. Both masks are
  binary; the optic disc mask should include the optic cup region.
* Optional physical spacing: ``spacing_y_mm,spacing_x_mm`` in original-image
  mm/pixel. Both columns must be present together.

Separate train, val, and test CSVs are recommended. A combined CSV may instead
have a ``split`` column, in which case callers must request one split. Old CSVs
with an unnamed pandas index are accepted, but image and mask paths must be
explicit. Use ``mask_column`` only when converting an existing manifest whose
mask column has a different name.
"""

from __future__ import annotations

import csv
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


_SPLITS = frozenset({"train", "val", "test"})


def _read_csv(manifest: str | Path) -> tuple[Path, list[str], list[dict[str, str]]]:
    path = Path(manifest).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        fields = [name.strip() for name in reader.fieldnames if name is not None]
        rows = [{(key or "").strip(): (value or "").strip()
                 for key, value in row.items() if key is not None}
                for row in reader]
    return path, fields, rows


def _resolve_path(raw: str, base: Path) -> Path:
    if not raw or not raw.strip():
        raise ValueError("Manifest contains an empty path")
    raw = raw.strip()
    # A drive or UNC path cannot be resolved correctly on a non-Windows host.
    if os.name != "nt" and PureWindowsPath(raw).drive:
        raise ValueError(f"Windows absolute path cannot be used on this host: {raw}")
    normalized = raw.replace("\\", "/")
    path = Path(normalized).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _base_dir(root: Path, subdir: str | Path | None) -> Path:
    return root if subdir is None else _resolve_path(str(subdir), root)


def _path_key(path: Path) -> str:
    # Case folding catches duplicate Windows paths even if checked on Linux.
    return str(path).replace("\\", "/").casefold()


def _select_rows(
    manifest: Path, fields: Sequence[str], rows: list[dict[str, str]], split: str | None
) -> list[dict[str, str]]:
    if split is not None and split not in _SPLITS:
        raise ValueError(f"split must be one of {sorted(_SPLITS)}")
    if "split" in fields:
        if split is None:
            raise ValueError("Manifest has a split column; pass split='train', 'val', or 'test'")
        rows = [row for row in rows if row.get("split", "").lower() == split]
    elif split is not None:
        # A separate split CSV is safe only when its filename names that split.
        parts = set(re.split(r"[^a-z0-9]+", manifest.stem.lower()))
        if split not in parts:
            raise ValueError(
                f"{manifest.name} has no split column and does not identify the {split!r} split"
            )
    if not rows:
        raise ValueError(f"No samples found in {manifest} for split {split!r}")
    return rows


@dataclass(frozen=True)
class ManifestRecord:
    image_path: Path
    mask_paths: tuple[Path, ...]
    spacing_mm: tuple[float, float] | None


class ManifestDataset(Dataset):
    """Load binary or overlapping optic disc / optic cup masks from a CSV.

    ``size`` is (height, width). Images are RGB float tensors in [0, 1] and
    masks are float tensors with shape [num_classes, height, width]. Default
    foreground is any nonzero grayscale pixel, supporting 0/1 and 0/255 PNGs.
    Geometry augmentation is applied jointly to images and masks only when
    requested: rotation within ±15 degrees, scaling within [0.8, 1.2], and
    independent horizontal/vertical flips, matching the paper's setting.
    Fundus data must provide separate OD and OC binary paths; this loader does
    not guess how a dataset encodes disc/cup in a single grayscale mask. Each
    sample includes original_size [H,W]. If physical spacing is in the CSV,
    it also includes resized_spacing [row_mm, column_mm] for output pixels.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        root: str | Path | None = None,
        split: str | None = None,
        image_dir: str | Path | None = None,
        mask_dir: str | Path | None = None,
        num_classes: int = 1,
        size: tuple[int, int] | int | None = (352, 352),
        augment: bool = False,
        mask_column: str | None = None,
        mask_threshold: float = 0.0,
        validate_files: bool = True,
    ) -> None:
        manifest_path, fields, rows = _read_csv(manifest)
        rows = _select_rows(manifest_path, fields, rows, split)
        if "image_path" not in fields:
            raise ValueError(
                f"{manifest_path.name} needs an image_path column; ImageId-only "
                "indexed CSVs must be converted to explicit image/mask paths"
            )
        if num_classes not in (1, 2):
            raise ValueError("num_classes must be 1 (binary) or 2 (OD/OC)")
        if num_classes == 1:
            selected_mask_column = mask_column or "mask_path"
            if selected_mask_column not in fields:
                raise ValueError(f"Missing mask column: {selected_mask_column}")
            mask_columns = (selected_mask_column,)
        else:
            if mask_column is not None:
                raise ValueError("mask_column is only valid for binary masks")
            mask_columns = ("od_mask_path", "oc_mask_path")
            for column in mask_columns:
                if column not in fields:
                    raise ValueError(f"Fundus manifest needs {column} column")
        if size is not None:
            if isinstance(size, int) and not isinstance(size, bool):
                size = (size, size)
            if (not isinstance(size, (tuple, list)) or len(size) != 2
                    or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
                           for v in size)):
                raise ValueError("size must be a positive (height, width) pair or None")
        if not np.isfinite(mask_threshold) or not 0 <= mask_threshold <= 1:
            raise ValueError("mask_threshold must be in [0, 1]")
        if augment and split in ("val", "test"):
            raise ValueError("Augmentation is only appropriate for the training split")

        root_path = manifest_path.parent if root is None else _resolve_path(str(root), Path.cwd())
        image_base = _base_dir(root_path, image_dir)
        mask_base = _base_dir(root_path, mask_dir)
        has_spacing = "spacing_y_mm" in fields or "spacing_x_mm" in fields
        if has_spacing and not {"spacing_y_mm", "spacing_x_mm"}.issubset(fields):
            raise ValueError("Spacing requires both spacing_y_mm and spacing_x_mm columns")
        records: list[ManifestRecord] = []
        seen: set[str] = set()
        for line_number, row in enumerate(rows, start=2):
            try:
                image = _resolve_path(row.get("image_path", ""), image_base)
                masks = tuple(_resolve_path(row.get(column, ""), mask_base)
                              for column in mask_columns)
            except ValueError as exc:
                raise ValueError(f"{manifest_path}:{line_number}: {exc}") from exc
            spacing_mm = None
            if has_spacing:
                try:
                    spacing_mm = (float(row["spacing_y_mm"]), float(row["spacing_x_mm"]))
                except (KeyError, ValueError) as exc:
                    raise ValueError(
                        f"{manifest_path}:{line_number}: spacing must contain two numbers"
                    ) from exc
                if not all(np.isfinite(value) and value > 0 for value in spacing_mm):
                    raise ValueError(
                        f"{manifest_path}:{line_number}: spacing must be positive and finite"
                    )
            key = _path_key(image)
            if key in seen:
                raise ValueError(f"Duplicate image path in {manifest_path}: {image}")
            seen.add(key)
            if validate_files:
                for file_path in (image, *masks):
                    if not file_path.is_file():
                        raise FileNotFoundError(
                            f"{manifest_path}:{line_number}: file not found: {file_path}"
                        )
            records.append(ManifestRecord(image, masks, spacing_mm))

        self.manifest = manifest_path
        self.records = records
        self.num_classes = num_classes
        self.size = size
        self.augment = augment
        self.mask_threshold = mask_threshold

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        with Image.open(record.image_path) as handle:
            image = handle.convert("RGB")
        original_height, original_width = image.height, image.width
        masks: list[Image.Image] = []
        for path in record.mask_paths:
            with Image.open(path) as handle:
                masks.append(handle.convert("L"))
        if any(mask.size != image.size for mask in masks):
            raise ValueError(f"Image/mask size mismatch for {record.image_path}")

        if self.augment:
            operations = []
            if random.random() < 0.5:
                operations.append(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                operations.append(Image.Transpose.FLIP_TOP_BOTTOM)
            for operation in operations:
                image = image.transpose(operation)
                masks = [mask.transpose(operation) for mask in masks]

            angle = math.radians(random.uniform(-15.0, 15.0))
            scale = random.uniform(0.8, 1.2)
            cosine = math.cos(angle) / scale
            sine = math.sin(angle) / scale
            center_x = (image.width - 1) / 2.0
            center_y = (image.height - 1) / 2.0
            coefficients = (
                cosine, sine, center_x - cosine * center_x - sine * center_y,
                -sine, cosine, center_y + sine * center_x - cosine * center_y,
            )
            image = image.transform(
                image.size, Image.Transform.AFFINE, coefficients,
                resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0),
            )
            masks = [mask.transform(
                mask.size, Image.Transform.AFFINE, coefficients,
                resample=Image.Resampling.NEAREST, fillcolor=0,
            ) for mask in masks]

        if self.size is not None:
            output_size = (self.size[1], self.size[0])
            image = image.resize(output_size, resample=Image.Resampling.BILINEAR)
            masks = [mask.resize(output_size, resample=Image.Resampling.NEAREST)
                     for mask in masks]

        image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mask_arrays = [np.asarray(mask, dtype=np.float32) / 255.0 > self.mask_threshold
                       for mask in masks]
        mask_array = np.stack(mask_arrays, axis=0).astype(np.float32)
        if self.num_classes == 2 and np.any(mask_array[1] > mask_array[0]):
            raise ValueError(
                f"Optic cup mask must be contained in the optic disc mask: {record.image_path}"
            )
        sample: dict[str, torch.Tensor | str] = {
            "image": torch.from_numpy(np.ascontiguousarray(image_array)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask_array)),
            "image_path": str(record.image_path),
            "original_size": torch.tensor((original_height, original_width), dtype=torch.int64),
        }
        if record.spacing_mm is not None:
            spacing_y, spacing_x = record.spacing_mm
            out_height, out_width = mask_array.shape[-2:]
            sample["resized_spacing"] = torch.tensor(
                (original_height * spacing_y / out_height,
                 original_width * spacing_x / out_width), dtype=torch.float32
            )
        if self.num_classes == 1:
            sample["mask_path"] = str(record.mask_paths[0])
        else:
            sample["od_mask_path"] = str(record.mask_paths[0])
            sample["oc_mask_path"] = str(record.mask_paths[1])
        return sample


def validate_disjoint_manifests(
    manifests: Mapping[str, str | Path],
    *,
    root: str | Path | None = None,
    image_dir: str | Path | None = None,
) -> None:
    """Raise if the same image occurs in two train/val/test manifest files.

    This checks identifiers without loading image files, so it can also audit
    indexed CSVs before the data are downloaded. Pass the same ``root`` and
    ``image_dir`` used by ``ManifestDataset``.
    """
    if not manifests:
        raise ValueError("No manifests supplied")
    owner: dict[str, str] = {}
    for split, manifest in manifests.items():
        if split not in _SPLITS:
            raise ValueError(f"Unknown split {split!r}")
        manifest_path, fields, rows = _read_csv(manifest)
        if "image_path" not in fields:
            raise ValueError(f"{manifest_path} has no image_path column")
        rows = _select_rows(manifest_path, fields, rows, split)
        root_path = manifest_path.parent if root is None else _resolve_path(str(root), Path.cwd())
        image_base = _base_dir(root_path, image_dir)
        within_split: set[str] = set()
        for row in rows:
            image = _resolve_path(row.get("image_path", ""), image_base)
            key = _path_key(image)
            if key in within_split:
                raise ValueError(f"Duplicate image in {split} manifest: {image}")
            within_split.add(key)
            previous = owner.get(key)
            if previous is not None:
                raise ValueError(
                    f"Image overlaps between {previous} and {split}: {image}"
                )
            owner[key] = split
