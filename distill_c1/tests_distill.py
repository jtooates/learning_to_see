"""
Unit tests for distillation module.

Tests:
1. Shape test: Forward pass produces correct shapes
2. Gradient test: Backprop computes gradients
3. Determinism: Fixed seed reproduces results
4. Loss toggling: Perceptual loss can be disabled
5. EMA: EMA updates work correctly
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
from pathlib import Path

from .text_encoder import build_text_encoder
from .decoder import build_decoder
from .losses import pixel_losses, tv_loss, rand_perc_loss, TinyRandNet, DistillationLoss
from .metrics import compute_psnr, compute_ssim
from .trainer import EMA


class TestShapes(unittest.TestCase):
    """Test that forward passes produce correct shapes."""

    def setUp(self):
        self.vocab_size = 32
        self.pad_id = 0
        self.batch_size = 4
        self.seq_len = 20
        self.emb_dim = 512
        self.H = 64
        self.W = 64

    def test_text_encoder_shape(self):
        """Test text encoder output shape."""
        encoder = build_text_encoder(vocab_size=self.vocab_size, pad_id=self.pad_id)
        encoder.eval()

        token_ids = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        e = encoder(token_ids, pad_id=self.pad_id)

        self.assertEqual(e.shape, (self.batch_size, self.emb_dim))
        self.assertTrue(torch.all(torch.isfinite(e)))

    def test_decoder_shape(self):
        """Test decoder output shape."""
        decoder = build_decoder()
        decoder.eval()

        e = torch.randn(self.batch_size, self.emb_dim)
        img = decoder(e)

        self.assertEqual(img.shape, (self.batch_size, 3, self.H, self.W))
        self.assertTrue(torch.all(torch.isfinite(img)))
        # Check tanh output range
        self.assertTrue(torch.all(img >= -1.0))
        self.assertTrue(torch.all(img <= 1.0))

    def test_end_to_end_shape(self):
        """Test end-to-end forward pass."""
        encoder = build_text_encoder(vocab_size=self.vocab_size, pad_id=self.pad_id)
        decoder = build_decoder()
        encoder.eval()
        decoder.eval()

        token_ids = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))

        with torch.no_grad():
            e = encoder(token_ids, pad_id=self.pad_id)
            img = decoder(e)

        self.assertEqual(img.shape, (self.batch_size, 3, self.H, self.W))


class TestGradients(unittest.TestCase):
    """Test that gradients flow through the model."""

    def setUp(self):
        self.vocab_size = 32
        self.pad_id = 0
        self.batch_size = 4
        self.seq_len = 20
        self.H = 64
        self.W = 64

    def test_encoder_gradients(self):
        """Test gradients flow through encoder."""
        encoder = build_text_encoder(vocab_size=self.vocab_size, pad_id=self.pad_id)

        token_ids = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        e = encoder(token_ids, pad_id=self.pad_id)

        loss = e.sum()
        loss.backward()

        # Check that gradients exist
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertTrue(torch.all(torch.isfinite(param.grad)), f"Non-finite gradient for {name}")

    def test_decoder_gradients(self):
        """Test gradients flow through decoder."""
        decoder = build_decoder()

        e = torch.randn(self.batch_size, 512, requires_grad=True)
        img = decoder(e)

        loss = img.sum()
        loss.backward()

        # Check that gradients exist
        for name, param in decoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertTrue(torch.all(torch.isfinite(param.grad)), f"Non-finite gradient for {name}")

    def test_loss_gradients(self):
        """Test gradients flow through loss functions."""
        pred = torch.randn(self.batch_size, 3, self.H, self.W, requires_grad=True)
        target = torch.randn(self.batch_size, 3, self.H, self.W)

        # Pixel losses
        pixel_dict = pixel_losses(pred, target)
        loss = pixel_dict['l1'] + pixel_dict['l2']
        loss.backward()
        self.assertIsNotNone(pred.grad)

        # TV loss
        pred.grad = None
        tv = tv_loss(pred)
        tv.backward()
        self.assertIsNotNone(pred.grad)


class TestDeterminism(unittest.TestCase):
    """Test that fixed seed produces deterministic results."""

    def setUp(self):
        self.vocab_size = 32
        self.pad_id = 0
        self.batch_size = 4
        self.seq_len = 20
        self.seed = 42

    def set_seed(self, seed):
        """Set all random seeds."""
        torch.manual_seed(seed)
        np.random.seed(seed)

    def test_encoder_determinism(self):
        """Test encoder produces same outputs with same seed."""
        token_ids = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))

        # First run
        self.set_seed(self.seed)
        encoder1 = build_text_encoder(vocab_size=self.vocab_size, pad_id=self.pad_id)
        encoder1.eval()
        with torch.no_grad():
            e1 = encoder1(token_ids, pad_id=self.pad_id)

        # Second run
        self.set_seed(self.seed)
        encoder2 = build_text_encoder(vocab_size=self.vocab_size, pad_id=self.pad_id)
        encoder2.eval()
        with torch.no_grad():
            e2 = encoder2(token_ids, pad_id=self.pad_id)

        # Should be identical
        torch.testing.assert_close(e1, e2)

    def test_decoder_determinism(self):
        """Test decoder produces same outputs with same seed."""
        e = torch.randn(self.batch_size, 512)

        # First run
        self.set_seed(self.seed)
        decoder1 = build_decoder()
        decoder1.eval()
        with torch.no_grad():
            img1 = decoder1(e)

        # Second run
        self.set_seed(self.seed)
        decoder2 = build_decoder()
        decoder2.eval()
        with torch.no_grad():
            img2 = decoder2(e)

        # Should be identical
        torch.testing.assert_close(img1, img2)


class TestLosses(unittest.TestCase):
    """Test loss functions."""

    def setUp(self):
        self.batch_size = 4
        self.H = 64
        self.W = 64

    def test_pixel_losses(self):
        """Test pixel losses compute correctly."""
        pred = torch.randn(self.batch_size, 3, self.H, self.W)
        target = torch.randn(self.batch_size, 3, self.H, self.W)

        pixel_dict = pixel_losses(pred, target)

        self.assertIn('l1', pixel_dict)
        self.assertIn('l2', pixel_dict)
        self.assertTrue(pixel_dict['l1'] > 0)
        self.assertTrue(pixel_dict['l2'] > 0)

        # Perfect reconstruction should have zero loss
        perfect_dict = pixel_losses(target, target)
        self.assertAlmostEqual(perfect_dict['l1'].item(), 0.0, places=5)
        self.assertAlmostEqual(perfect_dict['l2'].item(), 0.0, places=5)

    def test_tv_loss(self):
        """Test TV loss."""
        # Smooth image should have low TV
        smooth = torch.ones(self.batch_size, 3, self.H, self.W)
        tv_smooth = tv_loss(smooth)
        self.assertAlmostEqual(tv_smooth.item(), 0.0, places=5)

        # Noisy image should have high TV
        noisy = torch.randn(self.batch_size, 3, self.H, self.W)
        tv_noisy = tv_loss(noisy)
        self.assertTrue(tv_noisy > tv_smooth)

    def test_perceptual_loss(self):
        """Test random perceptual loss."""
        perc_net = TinyRandNet()

        # All parameters should be frozen
        for param in perc_net.parameters():
            self.assertFalse(param.requires_grad)

        pred = torch.randn(self.batch_size, 3, self.H, self.W)
        target = torch.randn(self.batch_size, 3, self.H, self.W)

        perc = rand_perc_loss(pred, target, perc_net)

        self.assertTrue(perc > 0)
        self.assertTrue(torch.isfinite(perc))

        # Perfect reconstruction should have zero loss
        perc_perfect = rand_perc_loss(target, target, perc_net)
        self.assertAlmostEqual(perc_perfect.item(), 0.0, places=5)

    def test_loss_toggling(self):
        """Test perceptual loss can be toggled on/off."""
        pred = torch.randn(self.batch_size, 3, self.H, self.W)
        target = torch.randn(self.batch_size, 3, self.H, self.W)

        # With perceptual loss
        loss_fn_with = DistillationLoss(use_perc=True)
        total_with, dict_with = loss_fn_with(pred, target)
        self.assertIn('perc', dict_with)

        # Without perceptual loss
        loss_fn_without = DistillationLoss(use_perc=False)
        total_without, dict_without = loss_fn_without(pred, target)
        self.assertNotIn('perc', dict_without)


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics."""

    def setUp(self):
        self.batch_size = 4
        self.H = 64
        self.W = 64

    def test_psnr(self):
        """Test PSNR computation."""
        pred = torch.randn(self.batch_size, 3, self.H, self.W) * 0.5
        target = torch.randn(self.batch_size, 3, self.H, self.W) * 0.5

        psnr = compute_psnr(pred, target)
        self.assertTrue(psnr > 0)
        self.assertTrue(np.isfinite(psnr))

        # Perfect reconstruction should have very high PSNR
        psnr_perfect = compute_psnr(target, target)
        self.assertTrue(psnr_perfect > 50)  # Should be very high

    def test_ssim(self):
        """Test SSIM computation."""
        pred = torch.randn(self.batch_size, 3, self.H, self.W) * 0.5
        target = torch.randn(self.batch_size, 3, self.H, self.W) * 0.5

        ssim = compute_ssim(pred, target)
        self.assertTrue(0 <= ssim <= 1)

        # Perfect reconstruction should have SSIM = 1
        ssim_perfect = compute_ssim(target, target)
        self.assertAlmostEqual(ssim_perfect, 1.0, places=3)


class TestEMA(unittest.TestCase):
    """Test Exponential Moving Average."""

    def test_ema_updates(self):
        """Test EMA updates correctly."""
        model = nn.Linear(10, 10)
        ema = EMA(model, decay=0.9)

        # Store initial shadow
        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Update model parameters
        for param in model.parameters():
            param.data += 0.1

        # Update EMA
        ema.update()

        # Shadow should have changed
        for name in ema.shadow:
            self.assertFalse(torch.allclose(ema.shadow[name], initial_shadow[name]))

    def test_ema_apply_restore(self):
        """Test EMA apply and restore."""
        model = nn.Linear(10, 10)
        ema = EMA(model, decay=0.9)

        # Store original parameters
        original_params = {name: param.data.clone() for name, param in model.named_parameters()}

        # Apply shadow
        ema.apply_shadow()

        # Parameters should have changed
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Should be equal to shadow
                torch.testing.assert_close(param.data, ema.shadow[name])

        # Restore
        ema.restore()

        # Parameters should be back to original
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.testing.assert_close(param.data, original_params[name])


class TestArchitecture(unittest.TestCase):
    """Test architectural details."""

    def test_no_checkerboard_artifacts(self):
        """Test that decoder uses upsample+conv (not transpose conv)."""
        decoder = build_decoder()

        # Check that no ConvTranspose2d is used
        for module in decoder.modules():
            self.assertNotIsInstance(module, nn.ConvTranspose2d,
                                   "Decoder should not use ConvTranspose2d (causes checkerboard artifacts)")

    def test_film_conditioning(self):
        """Test that FiLM layers exist in decoder."""
        decoder = build_decoder()

        # Count FiLM layers
        from .decoder import FiLM
        film_count = sum(1 for m in decoder.modules() if isinstance(m, FiLM))

        # Should have 3 FiLM layers (one per stage)
        self.assertEqual(film_count, 3, "Decoder should have 3 FiLM layers")

    def test_attention_layer(self):
        """Test that attention layer exists."""
        decoder = build_decoder()

        # Check for MHSA layer
        from .decoder import MHSA2d
        mhsa_count = sum(1 for m in decoder.modules() if isinstance(m, MHSA2d))

        # Should have 1 MHSA layer in stage 2
        self.assertEqual(mhsa_count, 1, "Decoder should have 1 MHSA layer")


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
