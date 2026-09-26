"""Prepare explicit OrthoSeg manifests from the preserved dataset CSV indexes.

This is an opt-in adapter. It never edits the original CSVs or source images.
``data_root`` is the user's biomedical image root; ``metadata_root`` is the
preserved ``Datasets/BioMedicalDataset`` directory containing CSV indexes.
The generated sidecars use absolute paths because DSB masks are derived under
``output_dir``, outside the source data root. Regenerate sidecars after moving
the data to another computer.

The DSB2018 adapter unions instance masks as the original loader did,
then stores a binary mask. The original loader also inverted some bright DSB
images with Otsu thresholding; this adapter deliberately does not apply that
image preprocessing. Thus it is not a bitwise replica of the old inference
pipeline. REFUGE and Drishti-GS have no indexed metadata here and are not
supported by this adapter. The original PolypSegData train CSV pools two
sources; request CVC-ClinicDB/train or Kvasir-SEG/train to select a single
source, or explicitly request PolypSegData-pooled/train to preserve the
old mixed set. None of these source indexes has a validation split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


SUPPORTED_DATASETS: dict[str, frozenset[str]] = {
    "BUSI": frozenset({"train", "val", "test"}),
    "COVID19": frozenset({"train", "test"}),
    "COVID19_2": frozenset({"test"}),
    "DSB2018": frozenset({"train", "val", "test"}),
    "ISIC2018": frozenset({"train", "val", "test"}),
    "MonuSeg2018": frozenset({"test"}),
    "PH2": frozenset({"test"}),
    "PolypSegData-pooled": frozenset({"train"}),
    "Kvasir-SEG": frozenset({"train"}),
    "STU": frozenset({"test"}),
    "CVC-300": frozenset({"test"}),
    "CVC-ClinicDB": frozenset({"train", "test"}),
    "CVC-ColonDB": frozenset({"test"}),
    "ETIS-LaribPolypDB": frozenset({"test"}),
    "Kvasir": frozenset({"test"}),
}

_POLYP_TARGETS = frozenset({
    "CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"
})
_POLYP_SOURCES = frozenset({
    "CVC-ClinicDB", "Kvasir-SEG", "PolypSegData-pooled"
})
_FIXED_FOLDERS = {
    "COVID19_2": ("images", "masks"),
    "MonuSeg2018": ("images", "masks"),
    "PH2": ("images", "masks"),
    "STU": ("image", "mask"),
}


def _metadata_path(dataset: str, split: str, metadata_root: Path) -> Path:
    if dataset == "DSB2018":
        return metadata_root / dataset / f"{split}_imageid_frame.csv"
    if split == "train" and dataset in _POLYP_SOURCES:
        return metadata_root / "PolypSegData" / "train_frame.csv"
    if dataset in _POLYP_TARGETS:
        return metadata_root / "PolypSegData" / f"{dataset}_test_frame.csv"
    return metadata_root / dataset / f"{split}_frame.csv"


def _read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Indexed CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Indexed CSV has no header: {path}")
        fields = {name.strip() for name in reader.fieldnames if name is not None}
        missing = required - fields
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = []
        for row in reader:
            normalized = {
                (key or "").strip(): (value or "").strip()
                for key, value in row.items() if key is not None
            }
            normalized["__source_line__"] = str(reader.line_num)
            rows.append(normalized)
    if not rows:
        raise ValueError(f"Indexed CSV is empty: {path}")
    return rows


def _safe_filename(raw: str, *, context: str) -> str:
    name = raw.strip()
    if (not name or name in {".", ".."} or "/" in name or "\\" in name
            or ":" in name or any(ord(char) < 32 for char in name)
            or Path(name).name != name):
        raise ValueError(f"{context}: expected a filename, got {raw!r}")
    return name


def _safe_image_id(raw: str, *, context: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", raw):
        raise ValueError(f"{context}: invalid ImageId {raw!r}")
    return raw


def _pair_paths(
    dataset: str, split: str, row: dict[str, str], data_root: Path, *, context: str
) -> tuple[Path, Path]:
    image_name = _safe_filename(row.get("image_path", ""), context=context)
    mask_column = "new_mask_path" if dataset == "ISIC2018" else "mask_path"
    mask_name = _safe_filename(row.get(mask_column, ""), context=context)
    image_stem, mask_stem = Path(image_name).stem, Path(mask_name).stem

    if dataset == "ISIC2018":
        expected_mask_stem = f"{image_stem}_segmentation"
    elif dataset == "BUSI":
        expected_mask_stem = f"{image_stem}_mask"
    elif dataset == "PH2":
        expected_mask_stem = f"{image_stem}_lesion"
    elif dataset == "STU":
        image_number = re.search(r"(\d+)$", image_stem)
        mask_number = re.search(r"(\d+)$", mask_stem)
        if image_number is None or mask_number is None or image_number.group(1) != mask_number.group(1):
            raise ValueError(f"{context}: STU image/mask IDs do not match")
        expected_mask_stem = mask_stem
    else:
        expected_mask_stem = image_stem
    if mask_stem != expected_mask_stem:
        raise ValueError(
            f"{context}: image {image_name!r} and mask {mask_name!r} do not match"
        )

    if dataset == "BUSI":
        category = image_name.split()[0]
        if category not in {"benign", "malignant", "normal"}:
            raise ValueError(f"{context}: unknown BUSI category in {image_name!r}")
        base = data_root / "BUSI" / category
        image_path, mask_path = base / "image" / image_name, base / "mask" / mask_name
    elif dataset == "COVID19":
        base = data_root / dataset / split
        image_path, mask_path = base / "images" / image_name, base / "masks" / mask_name
    elif dataset == "ISIC2018":
        base = data_root / dataset
        image_path = base / "ISIC2018_Task1-2_Training_Input" / image_name
        mask_path = base / "ISIC2018_Task1_Training_GroundTruth" / mask_name
    elif split == "train" and dataset in _POLYP_SOURCES:
        base = data_root / "PolypSegData" / "TrainDataset"
        image_path, mask_path = base / "images" / image_name, base / "masks" / mask_name
    elif dataset in _POLYP_TARGETS:
        base = data_root / "PolypSegData" / "TestDataset" / "TestDataset" / dataset
        image_path, mask_path = base / "images" / image_name, base / "masks" / mask_name
    else:
        image_folder, mask_folder = _FIXED_FOLDERS[dataset]
        base = data_root / dataset
        image_path = base / image_folder / image_name
        mask_path = base / mask_folder / mask_name
    return image_path.resolve(), mask_path.resolve()


def _image_size(path: Path, *, context: str) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"{context}: file not found: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            return image.size
    except OSError as exc:
        raise ValueError(f"{context}: unreadable image: {path}") from exc


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dsb_union(image: Path, mask_dir: Path, *, context: str) -> tuple[np.ndarray, list[Path]]:
    width, height = _image_size(image, context=context)
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"{context}: instance-mask folder not found: {mask_dir}")
    mask_paths = sorted(path for path in mask_dir.iterdir() if path.is_file())
    if not mask_paths:
        raise ValueError(f"{context}: no instance masks in {mask_dir}")
    union = np.zeros((height, width), dtype=np.uint8)
    for mask_path in mask_paths:
        if _image_size(mask_path, context=context) != (width, height):
            raise ValueError(f"{context}: image/mask dimensions differ: {mask_path}")
        with Image.open(mask_path) as handle:
            values = np.asarray(handle.convert("L"), dtype=np.uint8)
        # Equivalent to the original pixelwise maximum followed by binarization.
        np.maximum(union, values, out=union)
    return (union > 0).astype(np.uint8) * 255, mask_paths


def _write_csv(path: Path, rows: list[tuple[Path, Path]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_path", "mask_path"))
        for image_path, mask_path in rows:
            writer.writerow((str(image_path), str(mask_path)))


def _same_published_content(stage: Path, target: Path, manifest_name: str) -> bool:
    if not target.is_dir():
        return False
    for name in (manifest_name, "source_hashes.json"):
        staged_file, published_file = stage / name, target / name
        if not published_file.is_file() or staged_file.read_bytes() != published_file.read_bytes():
            return False
    staged_masks = stage / "derived_masks"
    published_masks = target / "derived_masks"
    if staged_masks.exists():
        if not published_masks.is_dir():
            return False
        staged_names = {path.name for path in staged_masks.iterdir() if path.is_file()}
        published_names = {path.name for path in published_masks.iterdir() if path.is_file()}
        if staged_names != published_names:
            return False
        return all(
            (staged_masks / name).read_bytes() == (published_masks / name).read_bytes()
            for name in staged_names
        )
    return not published_masks.exists()


def prepare_indexed_manifest(
    dataset: str,
    split: str,
    *,
    data_root: Path,
    metadata_root: Path,
    output_dir: Path,
) -> Path:
    """Build a validated sidecar manifest from one preserved dataset CSV.

    Returns ``output_dir/<dataset>_<split>/<split>_manifest.csv``. All image
    and mask paths in the CSV are absolute. Repeated calls with unchanged
    source files reuse the existing result; changed sources or altered output
    files raise rather than silently replacing an earlier preparation.

    This function does not enforce disjoint train/val/test indexes. Run
    ``validate_disjoint_manifests`` across the returned sidecars before use.
    In particular, the preserved BUSI validation and test CSVs overlap and
    must not be accepted together for a new experiment.
    """
    if dataset == "PolypSegData":
        raise ValueError(
            "PolypSegData/train is a mixed CVC-ClinicDB and Kvasir-SEG source. "
            "Select CVC-ClinicDB/train, Kvasir-SEG/train, or the explicitly "
            "pooled PolypSegData-pooled/train instead."
        )
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported indexed dataset {dataset!r}; REFUGE/Drishti-GS have no "
            "indexed CSV metadata. Supply explicit OD/OC manifests for fundus data."
        )
    if split not in SUPPORTED_DATASETS[dataset]:
        raise ValueError(f"{dataset} has no preserved {split!r} CSV split")
    data_root = Path(data_root).expanduser().resolve()
    metadata_root = Path(metadata_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Indexed data root not found: {data_root}")
    if not metadata_root.is_dir():
        raise FileNotFoundError(f"Indexed metadata root not found: {metadata_root}")
    metadata_path = _metadata_path(dataset, split, metadata_root)
    required = {"ImageId"} if dataset == "DSB2018" else {"image_path", "mask_path"}
    if dataset == "ISIC2018":
        required = {"image_path", "new_mask_path"}
    source_rows = _read_rows(metadata_path, required)
    if split == "train" and dataset in _POLYP_SOURCES:
        # The preserved train CSV combines 550 numeric CVC filenames with
        # 900 nonnumeric Kvasir filenames (including cju/cjy/ck2 prefixes).
        # Filter by the complete numeric-vs-nonnumeric partition, never a
        # single Kvasir prefix, and keep the pooled case explicit by name.
        if dataset != "PolypSegData-pooled":
            source_rows = [
                row for row in source_rows
                if Path(row.get("image_path", "")).stem.isdecimal()
                == (dataset == "CVC-ClinicDB")
            ]
        if not source_rows:
            raise ValueError(f"No {dataset} training rows found in {metadata_path}")

    target = output_dir / f"{dataset}_{split}"
    manifest_name = f"{split}_manifest.csv"
    manifest_path = target / manifest_name
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{dataset}_{split}_", dir=output_dir))
    try:
        output_rows: list[tuple[Path, Path]] = []
        fingerprints: dict[str, str] = {str(metadata_path.resolve()): _digest(metadata_path)}
        seen_images: set[str] = set()
        for row in source_rows:
            context = f"{metadata_path}:{row['__source_line__']}"
            if dataset == "DSB2018":
                image_id = _safe_image_id(row.get("ImageId", ""), context=context)
                base = data_root / "DSB2018" / "stage1_train" / image_id
                image_path = (base / "images" / f"{image_id}.png").resolve()
                mask_folder = base / "masks"
                union, source_masks = _dsb_union(image_path, mask_folder, context=context)
                staged_mask = stage / "derived_masks" / f"{image_id}.png"
                staged_mask.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(union).save(staged_mask)
                mask_path = target / "derived_masks" / f"{image_id}.png"
                for source_mask in source_masks:
                    fingerprints[str(source_mask.resolve())] = _digest(source_mask)
            else:
                image_path, mask_path = _pair_paths(
                    dataset, split, row, data_root, context=context
                )
                if _image_size(image_path, context=context) != _image_size(mask_path, context=context):
                    raise ValueError(f"{context}: image/mask dimensions differ")
                fingerprints[str(mask_path)] = _digest(mask_path)
            image_key = str(image_path).replace("\\", "/").casefold()
            if image_key in seen_images:
                raise ValueError(f"{context}: duplicate image path: {image_path}")
            seen_images.add(image_key)
            fingerprints[str(image_path)] = _digest(image_path)
            output_rows.append((image_path, mask_path))

        _write_csv(stage / manifest_name, output_rows)
        (stage / "source_hashes.json").write_text(
            json.dumps(
                {"dataset": dataset, "split": split,
                 "sources": [{"path": path, "sha256": digest}
                             for path, digest in sorted(fingerprints.items())]},
                ensure_ascii=False, indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            if _same_published_content(stage, target, manifest_name):
                return manifest_path
            raise FileExistsError(
                f"Prepared manifest already exists but source or output changed: {target}. "
                "Use a fresh output directory after reviewing the change."
            )
        stage.rename(target)
        return manifest_path
    finally:
        if stage.exists():
            shutil.rmtree(stage)
