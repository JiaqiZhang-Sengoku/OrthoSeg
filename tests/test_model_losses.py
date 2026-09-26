"""Contract tests for the paper-guided OrthoSeg model and objectives.

These tests use synthetic tensors only. They do not download data or weights.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from orthoseg import (
    CSTABlock,
    ConditionalGaussianMI,
    OrthoSeg,
    mid_weight,
    segmentation_loss,
    structure_equivariance_loss,
    texture_invariance_loss,
)


class TestOrthoSegModel(unittest.TestCase):
    def test_independent_encoders_and_odd_size_output(self) -> None:
        model = OrthoSeg(
            in_channels=3,
            num_classes=2,
            widths=(4, 8, 16, 32),
            latent_dim=8,
        )
        structure_parameter_ids = {id(p) for p in model.structure_encoder.parameters()}
        texture_parameter_ids = {id(p) for p in model.texture_encoder.parameters()}
        self.assertTrue(structure_parameter_ids)
        self.assertTrue(texture_parameter_ids)
        self.assertTrue(structure_parameter_ids.isdisjoint(texture_parameter_ids))

        image = torch.randn(2, 3, 17, 19)
        result = model(image, return_features=True)
        self.assertEqual(tuple(result.logits.shape), (2, 2, 17, 19))
        self.assertEqual(len(result.structure_features), 4)
        self.assertEqual(tuple(result.structure_latent.shape), (2, 8))
        self.assertEqual(tuple(result.texture_latent.shape), (2, 8))
        self.assertEqual(tuple(model(image[:1]).shape), (1, 2, 17, 19))

    def test_csta_routes_each_image_to_normalized_experts(self) -> None:
        block = CSTABlock(channels=4, num_experts=4)
        feature = torch.randn(2, 4, 5, 7)
        weights = block.routing_weights(feature)
        self.assertEqual(tuple(weights.shape), (2, 4))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
        self.assertTrue(torch.all(weights >= 0).item())
        self.assertEqual(tuple(block(feature).shape), tuple(feature.shape))


class TestOrthoSegLosses(unittest.TestCase):
    def test_mid_uses_only_strictly_off_diagonal_negatives(self) -> None:
        estimator = ConditionalGaussianMI(latent_dim=8, hidden_dim=8)
        latent = torch.zeros(3, 8)
        # Large diagonal values expose accidental inclusion of positive pairs
        # in the product-of-marginals term.
        pair_scores = torch.tensor(
            [[100.0, 1.0, 2.0], [3.0, 200.0, 4.0], [5.0, 6.0, 300.0]]
        )
        with patch.object(estimator, "log_prob_matrix", return_value=pair_scores) as scores:
            value = estimator.mid_loss(latent, latent)
        torch.testing.assert_close(value, torch.tensor(196.5))
        self.assertTrue(scores.call_args.kwargs["freeze_estimator"])
        with self.assertRaisesRegex(ValueError, "batch size >= 2"):
            estimator.mid_loss(latent[:1], latent[:1])

    def test_mid_alternating_steps_isolate_parameter_gradients(self) -> None:
        model = OrthoSeg(widths=(4, 8, 16, 32), latent_dim=8)
        estimator = ConditionalGaussianMI(latent_dim=8, hidden_dim=16)
        output = model(torch.randn(2, 3, 17, 19), return_features=True)

        estimator.auxiliary_loss(output.structure_latent, output.texture_latent).backward()
        self.assertTrue(any(p.grad is not None for p in estimator.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

        estimator.zero_grad(set_to_none=True)
        estimator.mid_loss(output.structure_latent, output.texture_latent).backward()
        self.assertTrue(all(p.grad is None for p in estimator.parameters()))
        self.assertIsNotNone(model.structure_projection[-1].weight.grad)
        self.assertIsNotNone(model.texture_projection[-1].weight.grad)

    def test_geometric_and_segmentation_losses_backpropagate(self) -> None:
        model = OrthoSeg(num_classes=2, widths=(4, 8, 16, 32), latent_dim=8)
        image = torch.randn(2, 3, 17, 19)
        flip = lambda tensor: torch.flip(tensor, dims=(-1,))
        original = model(image, return_features=True)
        transformed = model(flip(image), return_features=True)
        # Disc and cup masks may overlap, so use independent sigmoid channels.
        mask = torch.randint(0, 2, original.logits.shape).float()

        loss = (
            segmentation_loss(original.logits, mask, task="multilabel")
            + structure_equivariance_loss(
                original.structure_features, transformed.structure_features, flip
            )
            + texture_invariance_loss(original.texture_latent, transformed.texture_latent)
        )
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(model.head.weight.grad)
        self.assertIsNotNone(model.texture_projection[-1].weight.grad)
        self.assertTrue(torch.isfinite(model.head.weight.grad).all().item())

    def test_binary_loss_and_mid_schedule(self) -> None:
        logits = torch.randn(2, 1, 7, 9, requires_grad=True)
        mask = torch.randint(0, 2, logits.shape).float()
        loss = segmentation_loss(logits, mask, task="binary")
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertEqual(mid_weight(0), 1.0)
        self.assertEqual(mid_weight(200), 0.0)
        self.assertGreater(mid_weight(100), mid_weight(199))


if __name__ == "__main__":
    unittest.main()
