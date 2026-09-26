"""Synthetic, data-free tests for indexed manifest preparation."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from orthoseg.data import validate_disjoint_manifests
from orthoseg.indexed_data import prepare_indexed_manifest


def _csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _png(path: Path, values: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if values is None:
        values = np.zeros((4, 4), dtype=np.uint8)
    Image.fromarray(values).save(path)


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class IndexedPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.data = self.base / "private_data"
        self.metadata = self.base / "indexed_csv"
        self.output = self.base / "prepared"
        self.data.mkdir()
        self.metadata.mkdir()

    def _prepare(self, dataset: str, split: str) -> Path:
        return prepare_indexed_manifest(
            dataset, split, data_root=self.data,
            metadata_root=self.metadata, output_dir=self.output,
        )

    def test_isic_uses_corrected_mask_column_and_reuses_unchanged_output(self) -> None:
        source = self.data / "ISIC2018"
        image = source / "ISIC2018_Task1-2_Training_Input" / "ISIC_01.jpg"
        correct_mask = source / "ISIC2018_Task1_Training_GroundTruth" / "ISIC_01_segmentation.png"
        image.parent.mkdir(parents=True)
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(image)
        _png(correct_mask)
        _csv(self.metadata / "ISIC2018" / "train_frame.csv",
             ["", "image_path", "mask_path", "new_mask_path"], [
                 {"": "0", "image_path": image.name,
                  "mask_path": "unrelated_segmentation.png",
                  "new_mask_path": correct_mask.name}
             ])
        manifest = self._prepare("ISIC2018", "train")
        self.assertEqual(manifest.name, "train_manifest.csv")
        self.assertEqual(_manifest_rows(manifest)[0]["mask_path"], str(correct_mask.resolve()))
        self.assertEqual(self._prepare("ISIC2018", "train"), manifest)

        # A same-sized source change is still detected by provenance hashes.
        Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8)).save(image)
        with self.assertRaisesRegex(FileExistsError, "source or output changed"):
            self._prepare("ISIC2018", "train")

    def test_dsb_instances_are_unioned_to_a_binary_mask(self) -> None:
        image_id = "a" * 64
        sample = self.data / "DSB2018" / "stage1_train" / image_id
        image = sample / "images" / f"{image_id}.png"
        mask_a = np.zeros((4, 4), dtype=np.uint8)
        mask_b = np.zeros_like(mask_a)
        mask_a[1, 1] = 120
        mask_b[2, 2] = 255
        _png(image)
        _png(sample / "masks" / "a.png", mask_a)
        _png(sample / "masks" / "b.png", mask_b)
        _csv(self.metadata / "DSB2018" / "train_imageid_frame.csv",
             ["", "ImageId"], [{"": "0", "ImageId": image_id}])
        manifest = self._prepare("DSB2018", "train")
        row = _manifest_rows(manifest)[0]
        derived = Path(row["mask_path"])
        with Image.open(derived) as handle:
            result = np.asarray(handle)
        self.assertEqual(set(np.unique(result).tolist()), {0, 255})
        self.assertEqual(int(np.count_nonzero(result)), 2)
        self.assertEqual(self._prepare("DSB2018", "train"), manifest)

    def test_polyp_source_filter_is_explicit_and_complete(self) -> None:
        names = ["123.png", "cju123.png", "cjy123.png", "ck2abc.png"]
        source = self.data / "PolypSegData" / "TrainDataset"
        for name in names:
            _png(source / "images" / name)
            _png(source / "masks" / name)
        _csv(self.metadata / "PolypSegData" / "train_frame.csv",
             ["Image_Id", "image_path", "mask_path"], [
                 {"Image_Id": str(index), "image_path": name, "mask_path": name}
                 for index, name in enumerate(names)
             ])
        cvc = self._prepare("CVC-ClinicDB", "train")
        kvasir = self._prepare("Kvasir-SEG", "train")
        pooled = self._prepare("PolypSegData-pooled", "train")
        self.assertEqual(len(_manifest_rows(cvc)), 1)
        self.assertEqual(len(_manifest_rows(kvasir)), 3)
        self.assertEqual(len(_manifest_rows(pooled)), 4)
        with self.assertRaisesRegex(ValueError, "mixed"):
            self._prepare("PolypSegData", "train")
        with self.assertRaisesRegex(ValueError, "no preserved.*val"):
            self._prepare("Kvasir-SEG", "val")

        # The same dataset name with split=test must use the held-out target
        # index and TestDataset tree, never the mixed training CSV.
        test_root = (self.data / "PolypSegData" / "TestDataset" / "TestDataset"
                     / "CVC-ClinicDB")
        _png(test_root / "images" / "999.png")
        _png(test_root / "masks" / "999.png")
        _csv(self.metadata / "PolypSegData" / "CVC-ClinicDB_test_frame.csv",
             ["Image_Id", "image_path", "mask_path"], [
                 {"Image_Id": "0", "image_path": "999.png", "mask_path": "999.png"}
             ])
        held_out = self._prepare("CVC-ClinicDB", "test")
        self.assertEqual(len(_manifest_rows(held_out)), 1)
        self.assertIn("TestDataset", _manifest_rows(held_out)[0]["image_path"])

    def test_busi_overlap_remains_visible_to_split_audit(self) -> None:
        image_name = "benign (1).png"
        mask_name = "benign (1)_mask.png"
        root = self.data / "BUSI" / "benign"
        _png(root / "image" / image_name)
        _png(root / "mask" / mask_name)
        for split in ("val", "test"):
            _csv(self.metadata / "BUSI" / f"{split}_frame.csv",
                 ["", "image_path", "mask_path"], [
                     {"": "0", "image_path": image_name, "mask_path": mask_name}
                 ])
        val = self._prepare("BUSI", "val")
        test = self._prepare("BUSI", "test")
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_disjoint_manifests({"val": val, "test": test})

    def test_fixed_folder_pair_checks_dimensions(self) -> None:
        source = self.data / "MonuSeg2018"
        _png(source / "images" / "case.png")
        _png(source / "masks" / "case.png", np.zeros((3, 4), dtype=np.uint8))
        _csv(self.metadata / "MonuSeg2018" / "test_frame.csv",
             ["Image_Id", "image_path", "mask_path"], [
                 {"Image_Id": "0", "image_path": "case.png", "mask_path": "case.png"}
             ])
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            self._prepare("MonuSeg2018", "test")
        self.assertFalse((self.output / "MonuSeg2018_test").exists())

    def test_fundus_without_metadata_is_explicitly_unsupported(self) -> None:
        with self.assertRaisesRegex(ValueError, "no indexed CSV metadata"):
            self._prepare("REFUGE", "test")


if __name__ == "__main__":
    unittest.main()
