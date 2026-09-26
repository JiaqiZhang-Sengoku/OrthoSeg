"""Synthetic tests for the public manifest and metric interfaces."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from orthoseg.data import ManifestDataset, validate_disjoint_manifests
from orthoseg.metrics import binary_dice, binary_iou, evaluate_masks, hd95


def _write_manifest(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ManifestDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        (self.root / "masks").mkdir()
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[1:3, 1:3] = 255
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 255
        Image.fromarray(image).save(self.root / "images" / "a.png")
        Image.fromarray(mask).save(self.root / "masks" / "a.png")
        Image.fromarray(image).save(self.root / "images" / "b.png")
        Image.fromarray(mask).save(self.root / "masks" / "b.png")

    def test_binary_manifest_and_shared_augmentation(self) -> None:
        csv_path = self.root / "train.csv"
        _write_manifest(csv_path, ["image_path", "mask_path"], [
            {"image_path": r"images\a.png", "mask_path": "masks/a.png"}
        ])
        dataset = ManifestDataset(csv_path, split="train", size=None, augment=True)
        with patch("orthoseg.data.random.random", return_value=0.0), patch(
            "orthoseg.data.random.uniform", side_effect=[0.0, 1.0]
        ):
            sample = dataset[0]
        self.assertEqual(sample["image"].shape, (3, 4, 4))
        self.assertEqual(sample["mask"].shape, (1, 4, 4))
        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertTrue(torch.equal(sample["image"][0], sample["mask"][0]))
        self.assertEqual(len(dataset), 1)

    def test_resize_uses_nearest_neighbor_for_masks(self) -> None:
        csv_path = self.root / "train.csv"
        _write_manifest(csv_path, ["image_path", "mask_path"], [
            {"image_path": "images/a.png", "mask_path": "masks/a.png"}
        ])
        sample = ManifestDataset(csv_path, size=(6, 8))[0]
        self.assertEqual(sample["image"].shape, (3, 6, 8))
        self.assertEqual(sample["mask"].shape, (1, 6, 8))
        self.assertEqual(set(sample["mask"].unique().tolist()), {0.0, 1.0})

    def test_zero_one_png_and_resized_spacing(self) -> None:
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        Image.fromarray(mask).save(self.root / "masks" / "one.png")
        csv_path = self.root / "train.csv"
        _write_manifest(
            csv_path,
            ["image_path", "mask_path", "spacing_y_mm", "spacing_x_mm"],
            [{"image_path": "images/a.png", "mask_path": "masks/one.png",
              "spacing_y_mm": "0.2", "spacing_x_mm": "0.4"}],
        )
        sample = ManifestDataset(csv_path, size=(8, 8))[0]
        self.assertEqual(sample["mask"].sum().item(), 16.0)
        self.assertEqual(sample["original_size"].tolist(), [4, 4])
        self.assertTrue(torch.allclose(
            sample["resized_spacing"], torch.tensor([0.1, 0.2])
        ))

    def test_fundus_two_overlapping_channels(self) -> None:
        disc = np.zeros((4, 4), dtype=np.uint8)
        disc[1:3, 1:3] = 255
        cup = np.zeros((4, 4), dtype=np.uint8)
        cup[1, 1] = 255
        Image.fromarray(disc).save(self.root / "masks" / "disc.png")
        Image.fromarray(cup).save(self.root / "masks" / "cup.png")
        csv_path = self.root / "val.csv"
        _write_manifest(csv_path, ["image_path", "od_mask_path", "oc_mask_path"], [
            {"image_path": "images/a.png", "od_mask_path": "masks/disc.png",
             "oc_mask_path": "masks/cup.png"}
        ])
        sample = ManifestDataset(csv_path, split="val", num_classes=2, size=None)[0]
        self.assertEqual(sample["mask"].shape, (2, 4, 4))
        self.assertEqual(sample["mask"][0].sum().item(), 4.0)
        self.assertEqual(sample["mask"][1].sum().item(), 1.0)

    def test_fundus_rejects_cup_outside_disc(self) -> None:
        disc = np.zeros((4, 4), dtype=np.uint8)
        cup = np.zeros((4, 4), dtype=np.uint8)
        cup[1, 1] = 255
        Image.fromarray(disc).save(self.root / "masks" / "disc.png")
        Image.fromarray(cup).save(self.root / "masks" / "cup.png")
        csv_path = self.root / "val.csv"
        _write_manifest(csv_path, ["image_path", "od_mask_path", "oc_mask_path"], [
            {"image_path": "images/a.png", "od_mask_path": "masks/disc.png",
             "oc_mask_path": "masks/cup.png"}
        ])
        dataset = ManifestDataset(csv_path, split="val", num_classes=2, size=None)
        with self.assertRaisesRegex(ValueError, "Optic cup mask must be contained"):
            dataset[0]

    def test_split_filter_and_duplicate_rejection(self) -> None:
        csv_path = self.root / "combined.csv"
        _write_manifest(csv_path, ["image_path", "mask_path", "split"], [
            {"image_path": "images/a.png", "mask_path": "masks/a.png", "split": "train"},
            {"image_path": "images/b.png", "mask_path": "masks/b.png", "split": "val"},
        ])
        with self.assertRaisesRegex(ValueError, "split column"):
            ManifestDataset(csv_path)
        self.assertEqual(len(ManifestDataset(csv_path, split="val")), 1)
        duplicate = self.root / "train.csv"
        _write_manifest(duplicate, ["image_path", "mask_path"], [
            {"image_path": "images/a.png", "mask_path": "masks/a.png"},
            {"image_path": r"images\a.png", "mask_path": "masks/a.png"},
        ])
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            ManifestDataset(duplicate)

    def test_overlap_audit_rejects_busi_style_val_test_collision(self) -> None:
        val = self.root / "val_frame.csv"
        test = self.root / "test_frame.csv"
        # A leading blank index column occurs in the old pandas-generated CSVs.
        fields = ["", "image_path", "mask_path"]
        row = {"": "0", "image_path": "benign (1).png", "mask_path": "benign (1)_mask.png"}
        _write_manifest(val, fields, [row])
        _write_manifest(test, fields, [row])
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_disjoint_manifests({"val": val, "test": test}, root=self.root)


class MetricTests(unittest.TestCase):
    def test_exact_dice_and_iou(self) -> None:
        prediction = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        target = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        self.assertAlmostEqual(binary_dice(prediction, target), 0.5)
        self.assertAlmostEqual(binary_iou(prediction, target), 1 / 3)

    def test_hd95_is_surface_distance_and_honors_spacing(self) -> None:
        prediction = np.zeros((5, 6), dtype=np.uint8)
        target = np.zeros_like(prediction)
        prediction[2, 2] = 1
        target[2, 4] = 1
        self.assertEqual(hd95(prediction, target), 2.0)
        self.assertEqual(hd95(prediction, target, spacing=(1.0, 0.5)), 1.0)
        self.assertEqual(hd95(target, target), 0.0)

    def test_empty_mask_semantics(self) -> None:
        empty = np.zeros((4, 4), dtype=np.uint8)
        one = empty.copy()
        one[0, 0] = 1
        self.assertEqual(binary_dice(empty, empty), 1.0)
        self.assertEqual(binary_iou(empty, empty), 1.0)
        self.assertEqual(hd95(empty, empty), 0.0)
        self.assertEqual(binary_dice(one, empty), 0.0)
        self.assertEqual(binary_iou(one, empty), 0.0)
        self.assertTrue(math.isinf(hd95(one, empty)))

    def test_multilabel_macro_summary(self) -> None:
        prediction = torch.zeros((1, 2, 4, 4))
        target = torch.zeros_like(prediction)
        prediction[0, 0, 1, 1] = 1
        target[0, 0, 1, 1] = 1
        prediction[0, 1, 2, 2] = 1
        result = evaluate_masks(prediction, target, spacing=(0.2, 0.2))
        self.assertEqual(result["per_class"]["dice"], [1.0, 0.0])
        self.assertEqual(result["mean"]["dice"], 0.5)
        self.assertTrue(math.isinf(result["mean"]["hd95"]))
        self.assertEqual(result["unit"], "mm")

    def test_per_sample_spacing(self) -> None:
        prediction = np.zeros((2, 1, 5, 6), dtype=np.uint8)
        target = np.zeros_like(prediction)
        prediction[:, 0, 2, 2] = 1
        target[:, 0, 2, 4] = 1
        result = evaluate_masks(prediction, target, spacing=np.array([
            [1.0, 1.0], [1.0, 0.5]
        ]))
        self.assertAlmostEqual(result["mean"]["hd95"], 1.5)


if __name__ == "__main__":
    unittest.main()
