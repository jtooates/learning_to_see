"""Evaluation metrics for captioning."""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from sklearn.metrics import f1_score, confusion_matrix
import json


class CaptioningMetrics:
    """Comprehensive metrics for captioning evaluation."""

    def __init__(self, vocab):
        """Initialize metrics.

        Args:
            vocab: Vocabulary object (from dsl.tokens)
        """
        self.vocab = vocab
        self.reset()

        # Define attribute categories for per-attribute metrics
        self.colors = ['red', 'green', 'blue', 'yellow', 'white', 'gray']
        self.shapes = ['ball', 'cube', 'block']
        self.numbers = ['one', 'two', 'three', 'four', 'five']
        self.relations = ['left', 'right', 'on', 'in', 'front', 'of']

        # Create attribute sets for detection
        self.color_ids = [vocab.token_to_id.get(c, -1) for c in self.colors]
        self.shape_ids = [vocab.token_to_id.get(s, -1) for s in self.shapes]
        self.number_ids = [vocab.token_to_id.get(n, -1) for n in self.numbers]

    def reset(self):
        """Reset all metrics."""
        self.exact_matches = []
        self.token_correct = []
        self.token_total = []

        # Per-attribute predictions
        self.color_preds = []
        self.color_targets = []
        self.shape_preds = []
        self.shape_targets = []
        self.number_preds = []
        self.number_targets = []

        # Confusion tracking
        self.all_preds = []
        self.all_targets = []

    def update(self,
               pred_tokens: List[List[int]],
               target_tokens: List[List[int]],
               pred_texts: Optional[List[str]] = None,
               target_texts: Optional[List[str]] = None):
        """Update metrics with a batch of predictions.

        Args:
            pred_tokens: List of predicted token sequences
            target_tokens: List of target token sequences
            pred_texts: Optional list of predicted text strings
            target_texts: Optional list of target text strings
        """
        batch_size = len(pred_tokens)

        for i in range(batch_size):
            pred = pred_tokens[i]
            target = target_tokens[i]

            # Remove special tokens for comparison
            pred_clean = self._clean_sequence(pred)
            target_clean = self._clean_sequence(target)

            # Exact match
            exact = (pred_clean == target_clean)
            self.exact_matches.append(float(exact))

            # Token-level accuracy
            min_len = min(len(pred_clean), len(target_clean))
            if min_len > 0:
                correct = sum(p == t for p, t in zip(pred_clean[:min_len], target_clean[:min_len]))
                self.token_correct.append(correct)
                self.token_total.append(len(target_clean))

            # Extract attributes
            pred_colors = self._extract_tokens(pred_clean, self.color_ids)
            target_colors = self._extract_tokens(target_clean, self.color_ids)
            pred_shapes = self._extract_tokens(pred_clean, self.shape_ids)
            target_shapes = self._extract_tokens(target_clean, self.shape_ids)
            pred_numbers = self._extract_tokens(pred_clean, self.number_ids)
            target_numbers = self._extract_tokens(target_clean, self.number_ids)

            # Store for F1 computation (multi-label)
            self.color_preds.append(pred_colors)
            self.color_targets.append(target_colors)
            self.shape_preds.append(pred_shapes)
            self.shape_targets.append(target_shapes)
            self.number_preds.append(pred_numbers)
            self.number_targets.append(target_numbers)

            # Store all tokens for confusion matrix
            self.all_preds.extend(pred_clean)
            self.all_targets.extend(target_clean)

    def _clean_sequence(self, tokens: List[int]) -> List[int]:
        """Remove special tokens from sequence.

        Args:
            tokens: Token sequence

        Returns:
            Cleaned sequence without <BOS>, <EOS>, <PAD>
        """
        special = {self.vocab.bos_id, self.vocab.eos_id, self.vocab.pad_id}
        return [t for t in tokens if t not in special]

    def _extract_tokens(self, sequence: List[int], token_ids: List[int]) -> List[int]:
        """Extract specific tokens from sequence.

        Args:
            sequence: Token sequence
            token_ids: List of token IDs to extract

        Returns:
            List of matching token IDs
        """
        token_set = set(token_ids)
        return [t for t in sequence if t in token_set]

    def compute(self) -> Dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        # Exact match accuracy
        if self.exact_matches:
            metrics['exact_match'] = np.mean(self.exact_matches)
        else:
            metrics['exact_match'] = 0.0

        # Token accuracy
        if self.token_total:
            metrics['token_accuracy'] = sum(self.token_correct) / sum(self.token_total)
        else:
            metrics['token_accuracy'] = 0.0

        # Per-attribute F1 scores
        metrics['color_f1'] = self._compute_attribute_f1(self.color_preds, self.color_targets)
        metrics['shape_f1'] = self._compute_attribute_f1(self.shape_preds, self.shape_targets)
        metrics['number_f1'] = self._compute_attribute_f1(self.number_preds, self.number_targets)

        return metrics

    def _compute_attribute_f1(self,
                              preds: List[List[int]],
                              targets: List[List[int]]) -> float:
        """Compute F1 score for attribute extraction.

        Treats each sample as a multi-label classification problem.

        Args:
            preds: List of predicted attribute token lists
            targets: List of target attribute token lists

        Returns:
            F1 score
        """
        if not preds or not targets:
            return 0.0

        # Compute precision and recall
        tp = 0
        fp = 0
        fn = 0

        for pred_set, target_set in zip(preds, targets):
            pred_set = set(pred_set)
            target_set = set(target_set)

            tp += len(pred_set & target_set)
            fp += len(pred_set - target_set)
            fn += len(target_set - pred_set)

        # F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1

    def get_confusion_matrix(self, normalize: bool = False) -> Tuple[np.ndarray, List[str]]:
        """Compute confusion matrix for all tokens.

        Args:
            normalize: If True, normalize by true labels

        Returns:
            Tuple of (confusion_matrix, token_labels)
        """
        if not self.all_preds or not self.all_targets:
            return np.array([]), []

        # Get unique tokens
        all_tokens = sorted(set(self.all_preds + self.all_targets))
        token_labels = [self.vocab.id_to_token.get(t, f'<UNK_{t}>') for t in all_tokens]

        # Compute confusion matrix
        cm = confusion_matrix(
            self.all_targets,
            self.all_preds,
            labels=all_tokens
        )

        if normalize:
            cm = cm.astype('float')
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            cm = cm / row_sums

        return cm, token_labels

    def print_summary(self):
        """Print summary of metrics."""
        metrics = self.compute()

        print("\nCaptioning Metrics:")
        print(f"  Exact Match:     {metrics['exact_match']:.4f}")
        print(f"  Token Accuracy:  {metrics['token_accuracy']:.4f}")
        print(f"\nPer-Attribute F1:")
        print(f"  Color F1:        {metrics['color_f1']:.4f}")
        print(f"  Shape F1:        {metrics['shape_f1']:.4f}")
        print(f"  Number F1:       {metrics['number_f1']:.4f}")

    def save_json(self, path: str):
        """Save metrics to JSON file.

        Args:
            path: Output file path
        """
        metrics = self.compute()
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)


def evaluate_model(model, dataloader, vocab, device: str = 'cuda',
                   use_constraints: bool = True,
                   max_length: int = 32) -> CaptioningMetrics:
    """Evaluate model on a dataset.

    Args:
        model: Captioner model
        dataloader: DataLoader for evaluation
        vocab: Vocabulary object
        device: Device to run evaluation on
        use_constraints: If True, use FSM-constrained decoding
        max_length: Maximum generation length

    Returns:
        CaptioningMetrics object with results
    """
    from .decode import greedy_decode

    model.eval()
    metrics = CaptioningMetrics(vocab)

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            target_ids = batch['input_ids']

            # Generate captions
            pred_ids, pred_texts = greedy_decode(
                model=model,
                images=images,
                vocab=vocab,
                max_length=max_length,
                use_constraints=use_constraints
            )

            # Convert targets to list
            target_ids_list = target_ids.cpu().tolist()

            # Update metrics
            metrics.update(
                pred_tokens=pred_ids,
                target_tokens=target_ids_list,
                pred_texts=pred_texts
            )

    return metrics
