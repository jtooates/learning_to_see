"""Unit tests for the DSL components."""
import unittest
import json
from pathlib import Path

from dsl.tokens import Vocab
from dsl.parser import SceneParser, ParseError
from dsl.canonicalize import to_canonical, CanonicalizeError
from dsl.fsm import ConstrainedPolicy, DecodingState, State
from dsl.splits import (
    make_split_indices, enumerate_all_samples,
    ColorShapeHoldout, CountShapeHoldout, RelationHoldout
)


class TestVocab(unittest.TestCase):
    """Test the Vocab class."""

    def setUp(self):
        self.vocab = Vocab()

    def test_vocab_size(self):
        """Test vocabulary size is correct."""
        self.assertEqual(len(self.vocab), 32)

    def test_special_tokens(self):
        """Test special token IDs."""
        self.assertEqual(self.vocab.bos_id, 0)
        self.assertEqual(self.vocab.eos_id, 1)
        self.assertEqual(self.vocab.pad_id, 2)

    def test_encode_decode(self):
        """Test encoding and decoding."""
        text = "There is one red ball."
        encoded = self.vocab.encode(text, add_special=False)
        decoded = self.vocab.decode(encoded, skip_special=True)
        self.assertEqual(decoded, text)

    def test_encode_with_special(self):
        """Test encoding with special tokens."""
        text = "There is one red ball."
        encoded = self.vocab.encode(text, add_special=True)
        self.assertEqual(encoded[0], self.vocab.bos_id)
        self.assertEqual(encoded[-1], self.vocab.eos_id)

    def test_unknown_token(self):
        """Test that unknown tokens raise an error."""
        with self.assertRaises(ValueError):
            self.vocab.encode("unknown token")

    def test_pad_sequence(self):
        """Test sequence padding."""
        tokens = [3, 4, 5]
        padded = self.vocab.pad_sequence(tokens, max_length=5)
        self.assertEqual(len(padded), 5)
        self.assertEqual(padded[-2:], [self.vocab.pad_id, self.vocab.pad_id])


class TestParser(unittest.TestCase):
    """Test the SceneParser class."""

    def setUp(self):
        self.parser = SceneParser()

    def test_parse_count_singular(self):
        """Test parsing COUNT_SENT with singular."""
        text = "There is one red ball."
        scene = self.parser.parse(text)
        self.assertEqual(len(scene['objects']), 1)
        self.assertEqual(scene['objects'][0]['count'], 1)
        self.assertEqual(scene['objects'][0]['color'], 'red')
        self.assertEqual(scene['objects'][0]['shape'], 'ball')
        self.assertEqual(len(scene['relations']), 0)

    def test_parse_count_plural(self):
        """Test parsing COUNT_SENT with plural."""
        text = "There are three blue cubes."
        scene = self.parser.parse(text)
        self.assertEqual(len(scene['objects']), 1)
        self.assertEqual(scene['objects'][0]['count'], 3)
        self.assertEqual(scene['objects'][0]['color'], 'blue')
        self.assertEqual(scene['objects'][0]['shape'], 'cube')

    def test_parse_relation_left_of(self):
        """Test parsing REL_SENT with 'left of'."""
        text = "The red ball is left of the blue cube."
        scene = self.parser.parse(text)
        self.assertEqual(len(scene['objects']), 2)
        self.assertEqual(len(scene['relations']), 1)
        self.assertEqual(scene['relations'][0]['type'], 'left_of')
        self.assertEqual(scene['relations'][0]['src'], 'o1')
        self.assertEqual(scene['relations'][0]['dst'], 'o2')

    def test_parse_relation_on(self):
        """Test parsing REL_SENT with 'on'."""
        text = "The green block is on the yellow ball."
        scene = self.parser.parse(text)
        self.assertEqual(len(scene['objects']), 2)
        self.assertEqual(scene['relations'][0]['type'], 'on')

    def test_parse_relation_in_front_of(self):
        """Test parsing REL_SENT with 'in front of'."""
        text = "The blue cube is in front of the red block."
        scene = self.parser.parse(text)
        self.assertEqual(scene['relations'][0]['type'], 'in_front_of')

    def test_invalid_no_period(self):
        """Test that missing period raises error."""
        text = "There is one red ball"
        with self.assertRaises(ParseError):
            self.parser.parse(text)

    def test_invalid_plural_mismatch(self):
        """Test that singular/plural mismatch raises error."""
        text = "There is one red balls."
        with self.assertRaises(ParseError):
            self.parser.parse(text)

    def test_invalid_verb_mismatch(self):
        """Test that verb/number mismatch raises error."""
        text = "There are one red ball."
        with self.assertRaises(ParseError):
            self.parser.parse(text)

    def test_validate_method(self):
        """Test the validate method."""
        self.assertTrue(self.parser.validate("There is one red ball."))
        self.assertFalse(self.parser.validate("Invalid sentence."))


class TestCanonicalize(unittest.TestCase):
    """Test the canonicalize module."""

    def setUp(self):
        self.parser = SceneParser()

    def test_count_to_canonical(self):
        """Test COUNT_SENT canonicalization."""
        scene = {
            "canvas": {"W": 64, "H": 64, "bg": "white"},
            "objects": [{"id": "o1", "shape": "ball", "color": "red", "count": 1}],
            "relations": [],
            "constraints": {}
        }
        text = to_canonical(scene)
        self.assertEqual(text, "There is one red ball.")

    def test_count_plural_to_canonical(self):
        """Test COUNT_SENT plural canonicalization."""
        scene = {
            "canvas": {"W": 64, "H": 64, "bg": "white"},
            "objects": [{"id": "o1", "shape": "cube", "color": "blue", "count": 3}],
            "relations": [],
            "constraints": {}
        }
        text = to_canonical(scene)
        self.assertEqual(text, "There are three blue cubes.")

    def test_relation_to_canonical(self):
        """Test REL_SENT canonicalization."""
        scene = {
            "canvas": {"W": 64, "H": 64, "bg": "white"},
            "objects": [
                {"id": "o1", "shape": "ball", "color": "red", "count": 1},
                {"id": "o2", "shape": "cube", "color": "blue", "count": 1}
            ],
            "relations": [
                {"type": "left_of", "src": "o1", "dst": "o2"}
            ],
            "constraints": {}
        }
        text = to_canonical(scene)
        self.assertEqual(text, "The red ball is left of the blue cube.")

    def test_round_trip(self):
        """Test that parse and canonicalize are inverses."""
        test_sentences = [
            "There is one red ball.",
            "There are five yellow blocks.",
            "The red ball is left of the blue cube.",
            "The green block is on the yellow ball.",
            "The blue cube is in front of the red block."
        ]

        for text in test_sentences:
            scene = self.parser.parse(text)
            canonical = to_canonical(scene)
            self.assertEqual(canonical, text, f"Round-trip failed for: {text}")

    def test_invalid_missing_objects(self):
        """Test that missing objects field raises error."""
        scene = {"canvas": {}, "relations": []}
        with self.assertRaises(CanonicalizeError):
            to_canonical(scene)

    def test_invalid_count(self):
        """Test that invalid count raises error."""
        scene = {
            "objects": [{"id": "o1", "shape": "ball", "color": "red", "count": 10}],
            "relations": []
        }
        with self.assertRaises(CanonicalizeError):
            to_canonical(scene)


class TestFSM(unittest.TestCase):
    """Test the FSM constrained decoding."""

    def setUp(self):
        self.vocab = Vocab()
        self.policy = ConstrainedPolicy(self.vocab)

    def test_initial_state(self):
        """Test initial state."""
        state = self.policy.initial_state()
        self.assertEqual(state.fsm_state, State.START)

    def test_count_path(self):
        """Test FSM for COUNT_SENT path."""
        state = self.policy.initial_state()

        # Advance through: <BOS> There is one red ball . <EOS>
        tokens = ["<BOS>", "There", "is", "one", "red", "ball", ".", "<EOS>"]
        for token in tokens:
            token_id = self.vocab.get_id(token)
            allowed = self.policy.allowed_ids(state)
            self.assertIn(token_id, allowed,
                         f"Token '{token}' not allowed in state {state.fsm_state}")
            state = self.policy.advance(state, token_id)

        self.assertTrue(self.policy.is_terminal(state))

    def test_relation_path(self):
        """Test FSM for REL_SENT path."""
        state = self.policy.initial_state()

        # Advance through: <BOS> The red ball is left of the blue cube . <EOS>
        tokens = ["<BOS>", "The", "red", "ball", "is", "left", "of", "the", "blue", "cube", ".", "<EOS>"]
        for token in tokens:
            token_id = self.vocab.get_id(token)
            allowed = self.policy.allowed_ids(state)
            self.assertIn(token_id, allowed,
                         f"Token '{token}' not allowed in state {state.fsm_state}")
            state = self.policy.advance(state, token_id)

        self.assertTrue(self.policy.is_terminal(state))

    def test_invalid_token_rejected(self):
        """Test that invalid tokens raise an error."""
        state = self.policy.initial_state()
        # Try to use a color at the start (should fail)
        red_id = self.vocab.get_id("red")
        with self.assertRaises(ValueError):
            self.policy.advance(state, red_id)


class TestSplits(unittest.TestCase):
    """Test the splits module."""

    def test_enumerate_all_samples(self):
        """Test that enumeration produces correct number of samples."""
        samples = enumerate_all_samples()

        # COUNT_SENT: 4 colors × 3 shapes × 5 counts = 60
        # REL_SENT: 4 colors × 3 shapes × 4 relations × 4 colors × 3 shapes = 576
        # Total = 636
        expected_total = 60 + 576
        self.assertEqual(len(samples), expected_total)

    def test_random_split(self):
        """Test random split."""
        samples = enumerate_all_samples()
        splits = make_split_indices(samples, strategy='random', seed=42)

        self.assertIn('train', splits)
        self.assertIn('val', splits)
        self.assertIn('test', splits)

        # Check no overlap
        train_set = set(splits['train'])
        val_set = set(splits['val'])
        test_set = set(splits['test'])

        self.assertEqual(len(train_set & val_set), 0)
        self.assertEqual(len(train_set & test_set), 0)
        self.assertEqual(len(val_set & test_set), 0)

        # Check all indices covered
        all_indices = train_set | val_set | test_set
        self.assertEqual(len(all_indices), len(samples))

    def test_color_shape_holdout(self):
        """Test color-shape holdout split."""
        samples = enumerate_all_samples()
        holdout_pairs = [('yellow', 'cube')]
        splits = make_split_indices(
            samples,
            strategy='color_shape',
            holdout_pairs=holdout_pairs,
            seed=42
        )

        # Check that test set contains only yellow cubes
        for idx in splits['test']:
            sample = samples[idx]
            has_yellow_cube = any(
                obj['color'] == 'yellow' and obj['shape'] == 'cube'
                for obj in sample['objects']
            )
            self.assertTrue(has_yellow_cube)

        # Check that train/val don't contain yellow cubes
        for idx in splits['train'] + splits['val']:
            sample = samples[idx]
            has_yellow_cube = any(
                obj['color'] == 'yellow' and obj['shape'] == 'cube'
                for obj in sample['objects']
            )
            self.assertFalse(has_yellow_cube)

    def test_count_shape_holdout(self):
        """Test count-shape holdout split."""
        samples = enumerate_all_samples()
        holdout_pairs = [(5, 'ball')]
        splits = make_split_indices(
            samples,
            strategy='count_shape',
            holdout_pairs=holdout_pairs,
            seed=42
        )

        # Check that test set contains only 5 balls
        for idx in splits['test']:
            sample = samples[idx]
            has_five_balls = any(
                obj['count'] == 5 and obj['shape'] == 'ball'
                for obj in sample['objects']
            )
            self.assertTrue(has_five_balls)

    def test_relation_holdout(self):
        """Test relation holdout split."""
        samples = enumerate_all_samples()
        holdout_relations = ['in_front_of']
        splits = make_split_indices(
            samples,
            strategy='relation',
            holdout_relations=holdout_relations,
            seed=42
        )

        # Check that test set contains only in_front_of relations
        for idx in splits['test']:
            sample = samples[idx]
            if sample['relations']:
                self.assertEqual(sample['relations'][0]['type'], 'in_front_of')


class TestIntegration(unittest.TestCase):
    """Integration tests across components."""

    def setUp(self):
        self.vocab = Vocab()
        self.parser = SceneParser()

    def test_full_pipeline(self):
        """Test complete pipeline: text → scene → canonical → tokens."""
        text = "There are two green blocks."

        # Parse
        scene = self.parser.parse(text)

        # Canonicalize
        canonical = to_canonical(scene)
        self.assertEqual(canonical, text)

        # Tokenize
        tokens = self.vocab.encode(canonical, add_special=True)

        # Decode
        decoded = self.vocab.decode(tokens, skip_special=True)
        self.assertEqual(decoded, text)

    def test_all_valid_sentences_parse(self):
        """Test that all enumerated samples produce valid canonical text."""
        samples = enumerate_all_samples()

        for sample in samples:
            # Generate canonical text
            text = to_canonical(sample)

            # Parse it back
            scene = self.parser.parse(text)

            # Should round-trip
            text2 = to_canonical(scene)
            self.assertEqual(text, text2)


if __name__ == '__main__':
    unittest.main()
