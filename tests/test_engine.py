"""End-to-end contract checks using temporary synthetic images only.

These tests are shipped for future maintainers; they are not benchmark results.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from orthoseg.cli import build_parser
from orthoseg.engine import AffineView, evaluate_experiment, load_config, train_experiment
from orthoseg.losses import structure_equivariance_loss


class EngineContractTests(unittest.TestCase):
    def test_indexed_training_requires_same_source_for_validation(self) -> None:
        config = {
            "data": {"task": "binary", "indexed_datasets": {
                "train": "BUSI", "val": "ISIC2018", "test": "STU",
            }},
            "model": {"in_channels": 3, "num_classes": 1},
            "train": {},
        }
        with self.assertRaisesRegex(ValueError, "same source dataset"):
            train_experiment(config)

    def test_affine_consistency_excludes_zero_padded_border(self) -> None:
        view = AffineView(
            cosine=torch.tensor([1.25]), sine=torch.tensor([0.0]),
            horizontal=torch.tensor([1.0]), vertical=torch.tensor([1.0]),
        )
        feature = torch.zeros(1, 2, 5, 7)
        valid = view.valid_mask(feature)
        self.assertEqual(tuple(valid.shape), (1, 1, 5, 7))
        self.assertTrue(valid[0, 0, 2, 3].item())
        self.assertFalse(valid[0, 0, 0, 0].item())
        transformed = view(feature) + (~valid).float()
        loss = structure_equivariance_loss((feature,), (transformed,), view)
        self.assertEqual(loss.item(), 0.0)

    def test_cli_requires_command(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args(["evaluate", "--config", "config.json", "--checkpoint", "best.pt"])
        self.assertEqual(args.split, "test")

    def test_one_epoch_checkpoint_and_evaluation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data = root / "data"
            images = data / "images"
            masks = data / "masks"
            manifests = data / "manifests"
            for directory in (images, masks, manifests):
                directory.mkdir(parents=True)
            for index in range(4):
                image = np.zeros((16, 16, 3), dtype=np.uint8)
                mask = np.zeros((16, 16), dtype=np.uint8)
                image[4:12, 4:12] = 128 + index
                mask[4:12, 4:12] = 255
                Image.fromarray(image).save(images / f"{index}.png")
                Image.fromarray(mask).save(masks / f"{index}.png")

            for split, indices in (("train", (0, 1)), ("val", (2,)), ("test", (3,))):
                with (manifests / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("image_path", "mask_path"))
                    writer.writeheader()
                    for index in indices:
                        writer.writerow({
                            "image_path": f"images/{index}.png",
                            "mask_path": f"masks/{index}.png",
                        })

            config = {
                "data": {
                    "root": "data",
                    "train_manifest": "data/manifests/train.csv",
                    "val_manifest": "data/manifests/val.csv",
                    "test_manifest": "data/manifests/test.csv",
                    "task": "binary", "image_size": 16, "spacing": None,
                },
                "model": {
                    "in_channels": 3, "num_classes": 1,
                    "widths": [4, 8, 16, 32], "latent_dim": 8, "num_experts": 2,
                },
                "train": {
                    "epochs": 1, "batch_size": 2, "lr": 1e-4,
                    "aux_lr": 1e-4, "weight_decay": 0.0,
                    "alpha": 0.3, "beta": 1.0, "tau": 1.0,
                    "gamma": 0.9, "seed": 7, "num_workers": 0,
                    "output_dir": "runs/smoke",
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            loaded = load_config(config_path)
            checkpoint = train_experiment(loaded)
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((root / "runs" / "smoke" / "last.pt").is_file())
            self.assertTrue((root / "runs" / "smoke" / "history.jsonl").is_file())
            report = evaluate_experiment(loaded, checkpoint_path=checkpoint, split="test")
            summary = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(summary["unit"], "pixel")
            self.assertIn("hd95", summary["mean"])


if __name__ == "__main__":
    unittest.main()
