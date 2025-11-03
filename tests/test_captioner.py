"""Tests for captioner components."""
import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from captioner.encoder import ConvNeXtEncoder, ConvNeXtBlock, build_convnext_tiny
from captioner.decoder import AttentionGRUDecoder, BahdanauAttention, build_decoder
from captioner.model import Captioner, build_captioner
from captioner.decode import ConstrainedDecoder, greedy_decode, beam_search
from captioner.metrics import CaptioningMetrics
from captioner.augmentations import CaptionerAugmentation, get_train_augmentation
from dsl.tokens import Vocab


@pytest.fixture
def vocab():
    """Create vocab fixture."""
    return Vocab()


@pytest.fixture
def dummy_images():
    """Create dummy images (B=4, C=3, H=64, W=64)."""
    return torch.randn(4, 3, 64, 64)


@pytest.fixture
def dummy_targets(vocab):
    """Create dummy target sequences."""
    # Create sequences: <BOS> one green ball . <EOS> <PAD> ...
    batch_size = 4
    max_len = 10
    targets = torch.full((batch_size, max_len), vocab.pad_id, dtype=torch.long)

    # Sample sequence
    sample_seq = [vocab.bos_id, vocab.token_to_id['one'],
                  vocab.token_to_id['green'], vocab.token_to_id['ball'],
                  vocab.token_to_id['.'], vocab.eos_id]

    for i in range(batch_size):
        targets[i, :len(sample_seq)] = torch.tensor(sample_seq)

    return targets


class TestConvNeXtBlock:
    """Tests for ConvNeXt block."""

    def test_forward_shape(self):
        """Test forward pass shape."""
        block = ConvNeXtBlock(dim=64)
        x = torch.randn(2, 64, 16, 16)

        y = block(x)

        assert y.shape == x.shape, "Output shape should match input"

    def test_residual_connection(self):
        """Test residual connection exists."""
        block = ConvNeXtBlock(dim=64, drop_path=0.0)
        block.eval()

        x = torch.randn(1, 64, 16, 16)

        with torch.no_grad():
            y = block(x)

        # Output should be different from input (not identity)
        assert not torch.allclose(y, x), "Block should transform input"


class TestConvNeXtEncoder:
    """Tests for ConvNeXt encoder."""

    def test_build_encoder(self):
        """Test building encoder."""
        encoder = build_convnext_tiny()

        assert isinstance(encoder, ConvNeXtEncoder)
        assert encoder.final_dim == 256

    def test_forward_shapes(self, dummy_images):
        """Test forward pass output shapes."""
        encoder = build_convnext_tiny()

        grid_tokens, pooled = encoder(dummy_images)

        batch_size = dummy_images.size(0)
        assert grid_tokens.shape == (batch_size, 16, 256), "Grid tokens shape mismatch"
        assert pooled.shape == (batch_size, 256), "Pooled shape mismatch"

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        encoder = build_convnext_tiny()

        for batch_size in [1, 2, 8]:
            images = torch.randn(batch_size, 3, 64, 64)
            grid_tokens, pooled = encoder(images)

            assert grid_tokens.shape == (batch_size, 16, 256)
            assert pooled.shape == (batch_size, 256)


class TestBahdanauAttention:
    """Tests for Bahdanau attention."""

    def test_attention_output(self):
        """Test attention output shapes."""
        attention = BahdanauAttention(
            hidden_dim=512,
            encoder_dim=256,
            attention_dim=256
        )

        hidden = torch.randn(4, 512)
        encoder_out = torch.randn(4, 16, 256)

        context, weights = attention(hidden, encoder_out)

        assert context.shape == (4, 256), "Context shape mismatch"
        assert weights.shape == (4, 16), "Weights shape mismatch"

    def test_attention_weights_sum_to_one(self):
        """Test attention weights sum to 1."""
        attention = BahdanauAttention(
            hidden_dim=512,
            encoder_dim=256,
            attention_dim=256
        )

        hidden = torch.randn(2, 512)
        encoder_out = torch.randn(2, 16, 256)

        _, weights = attention(hidden, encoder_out)

        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums)), "Weights should sum to 1"


class TestAttentionGRUDecoder:
    """Tests for attention GRU decoder."""

    def test_build_decoder(self, vocab):
        """Test building decoder."""
        decoder = build_decoder(vocab_size=vocab.vocab_size)

        assert isinstance(decoder, AttentionGRUDecoder)
        assert decoder.vocab_size == vocab.vocab_size

    def test_forward_step(self, vocab):
        """Test single decoding step."""
        decoder = build_decoder(vocab_size=vocab.vocab_size)

        input_token = torch.tensor([vocab.bos_id, vocab.bos_id])
        hidden = torch.randn(1, 2, 512)
        encoder_out = torch.randn(2, 16, 256)

        logits, new_hidden, attn_weights = decoder.forward_step(
            input_token, hidden, encoder_out
        )

        assert logits.shape == (2, vocab.vocab_size), "Logits shape mismatch"
        assert new_hidden.shape == (1, 2, 512), "Hidden shape mismatch"
        assert attn_weights.shape == (2, 16), "Attention weights shape mismatch"

    def test_forward_training(self, vocab, dummy_targets):
        """Test forward pass for training."""
        decoder = build_decoder(vocab_size=vocab.vocab_size)

        batch_size = 4
        encoder_out = torch.randn(batch_size, 16, 256)
        pooled = torch.randn(batch_size, 256)

        logits = decoder(
            encoder_out=encoder_out,
            pooled=pooled,
            targets=dummy_targets,
            teacher_forcing_ratio=1.0
        )

        max_len = dummy_targets.size(1)
        assert logits.shape == (batch_size, max_len - 1, vocab.vocab_size)


class TestCaptioner:
    """Tests for full captioner model."""

    def test_build_captioner(self, vocab):
        """Test building captioner."""
        model = build_captioner(vocab_size=vocab.vocab_size)

        assert isinstance(model, Captioner)
        assert model.vocab_size == vocab.vocab_size

    def test_forward(self, vocab, dummy_images, dummy_targets):
        """Test forward pass."""
        model = build_captioner(vocab_size=vocab.vocab_size)

        logits, loss = model(
            images=dummy_images,
            targets=dummy_targets,
            teacher_forcing_ratio=1.0
        )

        batch_size = dummy_images.size(0)
        max_len = dummy_targets.size(1)

        assert logits.shape == (batch_size, max_len - 1, vocab.vocab_size)
        assert loss.item() > 0, "Loss should be positive"

    def test_encode(self, vocab, dummy_images):
        """Test encoding images."""
        model = build_captioner(vocab_size=vocab.vocab_size)

        grid_tokens, pooled = model.encode(dummy_images)

        batch_size = dummy_images.size(0)
        assert grid_tokens.shape == (batch_size, 16, 256)
        assert pooled.shape == (batch_size, 256)

    def test_decode_step(self, vocab, dummy_images):
        """Test single decode step."""
        model = build_captioner(vocab_size=vocab.vocab_size)

        # Encode
        grid_tokens, pooled = model.encode(dummy_images)

        # Initialize decoder state
        hidden = model.init_decoder_state(pooled)

        # Decode step
        batch_size = dummy_images.size(0)
        input_token = torch.full((batch_size,), vocab.bos_id, dtype=torch.long)

        logits, new_hidden, attn_weights = model.decode_step(
            input_token=input_token,
            hidden=hidden,
            encoder_out=grid_tokens
        )

        assert logits.shape == (batch_size, vocab.vocab_size)
        assert new_hidden.shape == (1, batch_size, 512)


class TestConstrainedDecoder:
    """Tests for constrained decoding."""

    def test_greedy_decode(self, vocab, dummy_images):
        """Test greedy decoding."""
        model = build_captioner(vocab_size=vocab.vocab_size)
        model.eval()

        token_ids, texts = greedy_decode(
            model=model,
            images=dummy_images,
            vocab=vocab,
            max_length=32,
            use_constraints=False
        )

        batch_size = dummy_images.size(0)
        assert len(token_ids) == batch_size
        assert len(texts) == batch_size

        # Check sequences start with BOS
        for seq in token_ids:
            assert seq[0] == vocab.bos_id

    def test_greedy_decode_with_constraints(self, vocab, dummy_images):
        """Test greedy decoding with FSM constraints."""
        model = build_captioner(vocab_size=vocab.vocab_size)
        model.eval()

        token_ids, texts = greedy_decode(
            model=model,
            images=dummy_images[:1],  # Single image for speed
            vocab=vocab,
            max_length=32,
            use_constraints=True
        )

        assert len(token_ids) == 1
        assert len(texts) == 1

    def test_beam_search(self, vocab, dummy_images):
        """Test beam search."""
        model = build_captioner(vocab_size=vocab.vocab_size)
        model.eval()

        token_ids, texts = beam_search(
            model=model,
            images=dummy_images[:1],  # Single image
            vocab=vocab,
            beam_size=3,
            max_length=32,
            use_constraints=False
        )

        assert len(token_ids) == 1
        assert len(texts) == 1


class TestCaptioningMetrics:
    """Tests for captioning metrics."""

    def test_exact_match(self, vocab):
        """Test exact match metric."""
        metrics = CaptioningMetrics(vocab)

        # Perfect prediction
        pred = [[vocab.bos_id, vocab.token_to_id['one'], vocab.eos_id]]
        target = [[vocab.bos_id, vocab.token_to_id['one'], vocab.eos_id]]

        metrics.update(pred, target)
        results = metrics.compute()

        assert results['exact_match'] == 1.0

    def test_token_accuracy(self, vocab):
        """Test token accuracy."""
        metrics = CaptioningMetrics(vocab)

        # After removing special tokens: "one", "red" vs "one", "blue"
        # 1/2 correct = 0.5
        pred = [[vocab.bos_id, vocab.token_to_id['one'], vocab.token_to_id['red'], vocab.eos_id]]
        target = [[vocab.bos_id, vocab.token_to_id['one'], vocab.token_to_id['blue'], vocab.eos_id]]

        metrics.update(pred, target)
        results = metrics.compute()

        # Should be 1/2 = 0.5
        assert results['token_accuracy'] == 0.5

    def test_attribute_f1(self, vocab):
        """Test attribute F1 scores."""
        metrics = CaptioningMetrics(vocab)

        # Correct color and shape
        pred = [[vocab.bos_id, vocab.token_to_id['red'], vocab.token_to_id['ball'], vocab.eos_id]]
        target = [[vocab.bos_id, vocab.token_to_id['red'], vocab.token_to_id['ball'], vocab.eos_id]]

        metrics.update(pred, target)
        results = metrics.compute()

        assert results['color_f1'] == 1.0
        assert results['shape_f1'] == 1.0


class TestAugmentations:
    """Tests for data augmentations."""

    def test_train_augmentation(self):
        """Test training augmentation."""
        aug = get_train_augmentation(image_size=64, strong=True)

        image = torch.rand(3, 64, 64)
        augmented = aug(image)

        assert augmented.shape == (3, 64, 64)
        assert augmented.min() >= 0.0
        assert augmented.max() <= 1.0

    def test_augmentation_deterministic(self):
        """Test augmentation is different each time."""
        aug = get_train_augmentation(image_size=64, strong=True)

        image = torch.rand(3, 64, 64)
        aug1 = aug(image)
        aug2 = aug(image)

        # Should be different due to randomness
        assert not torch.allclose(aug1, aug2)

    def test_cutout(self):
        """Test cutout augmentation."""
        aug = CaptionerAugmentation(
            image_size=64,
            cutout_holes=1,
            cutout_size=8
        )

        image = torch.ones(3, 64, 64)
        augmented = aug(image)

        # Should have some pixels at 0.5 (cutout fill value)
        assert (augmented == 0.5).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
