"""Constrained decoding with FSM for grammar-compliant generation."""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

from dsl.fsm import ConstrainedPolicy


@dataclass
class BeamHypothesis:
    """Single beam hypothesis during beam search."""
    tokens: List[int]
    score: float
    hidden: torch.Tensor
    fsm_state: any  # DecodingState from FSM


class ConstrainedDecoder:
    """Decoder with FSM-constrained generation."""

    def __init__(self, model, vocab, max_length: int = 32):
        """Initialize constrained decoder.

        Args:
            model: Captioner model
            vocab: Vocabulary object (from dsl.tokens)
            max_length: Maximum generation length
        """
        self.model = model
        self.vocab = vocab
        self.max_length = max_length
        self.bos_id = vocab.bos_id
        self.eos_id = vocab.eos_id
        self.pad_id = vocab.pad_id

    def greedy_decode(self,
                      images: torch.Tensor,
                      use_constraints: bool = True) -> Tuple[List[List[int]], List[str]]:
        """Greedy decoding with optional FSM constraints.

        Args:
            images: Input images (B, 3, 64, 64)
            use_constraints: If True, use FSM to constrain generation

        Returns:
            Tuple of:
            - token_ids: List of token ID sequences (B, varying lengths)
            - texts: List of decoded text strings
        """
        self.model.eval()
        batch_size = images.size(0)

        with torch.no_grad():
            # Encode images
            grid_tokens, pooled = self.model.encode(images)

            # Initialize decoder states
            hidden = self.model.init_decoder_state(pooled)

            # Initialize FSM for each sample if using constraints
            if use_constraints:
                fsm_policies = [ConstrainedPolicy(self.vocab) for _ in range(batch_size)]
                fsm_states = [policy.initial_state() for policy in fsm_policies]
            else:
                fsm_policies = [None] * batch_size
                fsm_states = [None] * batch_size

            # Storage for generated sequences
            generated = [[self.bos_id] for _ in range(batch_size)]
            finished = [False] * batch_size

            # Start with <BOS> token
            input_token = torch.full((batch_size,), self.bos_id, dtype=torch.long, device=images.device)

            # Generate tokens
            for step in range(self.max_length):
                # Decode step
                logits, hidden, _ = self.model.decode_step(
                    input_token=input_token,
                    hidden=hidden,
                    encoder_out=grid_tokens
                )

                # Apply FSM constraints if enabled
                if use_constraints:
                    for i in range(batch_size):
                        if not finished[i] and fsm_policies[i] is not None:
                            allowed = fsm_policies[i].allowed_ids(fsm_states[i])
                            # Mask out disallowed tokens
                            mask = torch.ones(self.vocab.vocab_size, device=logits.device) * float('-inf')
                            mask[allowed] = 0
                            logits[i] = logits[i] + mask

                # Greedy selection
                next_token = logits.argmax(dim=1)

                # Update sequences
                for i in range(batch_size):
                    if not finished[i]:
                        token_id = next_token[i].item()
                        generated[i].append(token_id)

                        # Update FSM state
                        if use_constraints and fsm_policies[i] is not None:
                            fsm_states[i] = fsm_policies[i].advance(fsm_states[i], token_id)

                        # Check for EOS
                        if token_id == self.eos_id:
                            finished[i] = True

                # Stop if all sequences finished
                if all(finished):
                    break

                # Prepare next input
                input_token = next_token

        # Decode to text
        texts = [self.vocab.decode(tokens) for tokens in generated]

        return generated, texts

    def beam_search(self,
                    images: torch.Tensor,
                    beam_size: int = 5,
                    use_constraints: bool = True) -> Tuple[List[List[int]], List[str]]:
        """Beam search decoding with optional FSM constraints.

        Args:
            images: Input images (B, 3, 64, 64)
            beam_size: Beam size
            use_constraints: If True, use FSM to constrain generation

        Returns:
            Tuple of:
            - token_ids: List of best token ID sequences (B sequences)
            - texts: List of decoded text strings
        """
        self.model.eval()
        batch_size = images.size(0)

        if batch_size != 1:
            # For simplicity, only support batch_size=1 for beam search
            # Run multiple single-image beam searches
            all_tokens = []
            all_texts = []
            for i in range(batch_size):
                tokens, texts = self.beam_search(
                    images[i:i+1],
                    beam_size=beam_size,
                    use_constraints=use_constraints
                )
                all_tokens.extend(tokens)
                all_texts.extend(texts)
            return all_tokens, all_texts

        with torch.no_grad():
            # Encode image
            grid_tokens, pooled = self.model.encode(images)

            # Initialize decoder state
            hidden = self.model.init_decoder_state(pooled)

            # Initialize FSM
            if use_constraints:
                fsm_policy = ConstrainedPolicy(self.vocab)
            else:
                fsm_policy = None

            # Initialize beams
            beams = [BeamHypothesis(
                tokens=[self.bos_id],
                score=0.0,
                hidden=hidden,
                fsm_state=fsm_policy.state if fsm_policy else None
            )]

            completed = []

            # Beam search
            for step in range(self.max_length):
                candidates = []

                for beam in beams:
                    # Skip if beam is complete
                    if beam.tokens[-1] == self.eos_id:
                        completed.append(beam)
                        continue

                    # Prepare input
                    input_token = torch.tensor([beam.tokens[-1]], dtype=torch.long, device=images.device)

                    # Decode step
                    logits, new_hidden, _ = self.model.decode_step(
                        input_token=input_token,
                        hidden=beam.hidden,
                        encoder_out=grid_tokens
                    )

                    # Log probabilities
                    log_probs = F.log_softmax(logits, dim=1).squeeze(0)

                    # Apply FSM constraints if enabled
                    if use_constraints and fsm_policy is not None:
                        allowed = fsm_policy.allowed_ids(beam.fsm_state)
                        # Mask out disallowed tokens
                        mask = torch.ones_like(log_probs) * float('-inf')
                        mask[allowed] = 0
                        log_probs = log_probs + mask

                    # Get top-k candidates
                    top_log_probs, top_indices = log_probs.topk(beam_size)

                    # Create new hypotheses
                    for log_prob, token_id in zip(top_log_probs, top_indices):
                        # Update FSM state
                        if use_constraints and fsm_policy is not None:
                            new_fsm_state = fsm_policy.advance(beam.fsm_state, token_id.item())
                        else:
                            new_fsm_state = None

                        # Create new beam
                        new_beam = BeamHypothesis(
                            tokens=beam.tokens + [token_id.item()],
                            score=beam.score + log_prob.item(),
                            hidden=new_hidden,
                            fsm_state=new_fsm_state
                        )
                        candidates.append(new_beam)

                # Keep top beams
                if candidates:
                    candidates.sort(key=lambda x: x.score / len(x.tokens), reverse=True)
                    beams = candidates[:beam_size]
                else:
                    break

                # Stop if all beams are complete
                if all(beam.tokens[-1] == self.eos_id for beam in beams):
                    completed.extend(beams)
                    break

            # Get best hypothesis
            if completed:
                best = max(completed, key=lambda x: x.score / len(x.tokens))
            else:
                best = max(beams, key=lambda x: x.score / len(x.tokens))

            # Decode to text
            text = self.vocab.decode(best.tokens)

            return [best.tokens], [text]


def greedy_decode(model, images: torch.Tensor, vocab,
                  max_length: int = 32,
                  use_constraints: bool = True) -> Tuple[List[List[int]], List[str]]:
    """Greedy decoding helper function.

    Args:
        model: Captioner model
        images: Input images (B, 3, 64, 64)
        vocab: Vocabulary object
        max_length: Maximum generation length
        use_constraints: If True, use FSM to constrain generation

    Returns:
        Tuple of (token_ids, texts)
    """
    decoder = ConstrainedDecoder(model, vocab, max_length)
    return decoder.greedy_decode(images, use_constraints)


def beam_search(model, images: torch.Tensor, vocab,
                beam_size: int = 5,
                max_length: int = 32,
                use_constraints: bool = True) -> Tuple[List[List[int]], List[str]]:
    """Beam search helper function.

    Args:
        model: Captioner model
        images: Input images (B, 3, 64, 64)
        vocab: Vocabulary object
        beam_size: Beam size
        max_length: Maximum generation length
        use_constraints: If True, use FSM to constrain generation

    Returns:
        Tuple of (token_ids, texts)
    """
    decoder = ConstrainedDecoder(model, vocab, max_length)
    return decoder.beam_search(images, beam_size, use_constraints)
