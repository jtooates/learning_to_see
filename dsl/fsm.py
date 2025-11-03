"""Finite-state machine for constrained decoding in the scene DSL."""
from typing import List, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto


class State(Enum):
    """FSM states for the scene DSL grammar."""
    START = auto()

    # COUNT_SENT path: "There (is|are) NUMBER COLOR SHAPE(s)? ."
    THERE = auto()
    COUNT_VERB = auto()  # is/are
    COUNT_NUMBER = auto()
    COUNT_COLOR = auto()
    COUNT_SHAPE = auto()
    COUNT_MAYBE_PLURAL = auto()  # might need 's'

    # REL_SENT path: "The COLOR SHAPE (is|are) REL the COLOR SHAPE ."
    THE1 = auto()  # First "The"
    REL_COLOR1 = auto()
    REL_SHAPE1 = auto()
    REL_VERB = auto()  # is/are
    REL_TYPE = auto()  # left/right/on/in
    REL_OF = auto()  # "of" after left/right
    REL_FRONT = auto()  # "front" after "in"
    REL_OF2 = auto()  # "of" after "front"
    THE2 = auto()  # Second "the"
    REL_COLOR2 = auto()
    REL_SHAPE2 = auto()

    PERIOD = auto()
    END = auto()


@dataclass
class DecodingState:
    """State for constrained decoding, tracking FSM state and context."""
    fsm_state: State
    # Track context needed for validation
    count_value: Optional[int] = None  # For plural agreement
    in_relation_of: bool = False  # Are we in "left of" or "right of"?
    in_relation_front: bool = False  # Are we in "in front of"?


class ConstrainedPolicy:
    """Constrained decoding policy using FSM to restrict valid next tokens."""

    def __init__(self, vocab):
        """Initialize policy with vocabulary.

        Args:
            vocab: Vocab instance for token lookup
        """
        self.vocab = vocab

        # Build token ID sets for each category
        self.colors = {"red", "green", "blue", "yellow"}
        self.shapes = {"ball", "cube", "block"}
        self.shapes_plural = {"balls", "cubes", "blocks"}
        self.numbers = {"one", "two", "three", "four", "five"}
        self.number_to_count = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

        # Get token IDs
        self.color_ids = {vocab.get_id(c) for c in self.colors}
        self.shape_ids = {vocab.get_id(s) for s in self.shapes}
        self.shape_plural_ids = {vocab.get_id(s) for s in self.shapes_plural}
        self.number_ids = {vocab.get_id(n) for n in self.numbers}

        self.there_id = vocab.get_id("There")
        self.the_id = vocab.get_id("the")
        self.the_cap_id = vocab.get_id("The")  # Capitalized "The" for REL_SENT
        self.is_id = vocab.get_id("is")
        self.are_id = vocab.get_id("are")
        self.left_id = vocab.get_id("left")
        self.right_id = vocab.get_id("right")
        self.on_id = vocab.get_id("on")
        self.in_id = vocab.get_id("in")
        self.of_id = vocab.get_id("of")
        self.front_id = vocab.get_id("front")
        self.period_id = vocab.get_id(".")
        self.bos_id = vocab.bos_id
        self.eos_id = vocab.eos_id

    def initial_state(self) -> DecodingState:
        """Return the initial FSM state."""
        return DecodingState(fsm_state=State.START)

    def allowed_ids(self, state: DecodingState) -> List[int]:
        """Get list of allowed token IDs for the current state.

        Args:
            state: Current decoding state

        Returns:
            List of valid token IDs
        """
        s = state.fsm_state
        allowed = set()

        if s == State.START:
            # Can start with <BOS>, "There", or "The" (capitalized)
            allowed = {self.bos_id, self.there_id, self.the_cap_id}

        elif s == State.THERE:
            # After "There", must have "is" or "are"
            allowed = {self.is_id, self.are_id}

        elif s == State.COUNT_VERB:
            # After is/are, need a number
            allowed = self.number_ids

        elif s == State.COUNT_NUMBER:
            # After number, need a color
            allowed = self.color_ids

        elif s == State.COUNT_COLOR:
            # After color, need a shape (singular or plural depending on count)
            if state.count_value and state.count_value > 1:
                # Plural shapes
                allowed = self.shape_plural_ids
            else:
                # Singular shapes
                allowed = self.shape_ids

        elif s == State.COUNT_SHAPE:
            # After shape, go to period
            allowed = {self.period_id}

        elif s == State.THE1:
            # After first "The", need a color
            allowed = self.color_ids

        elif s == State.REL_COLOR1:
            # After first color, need a shape
            allowed = self.shape_ids

        elif s == State.REL_SHAPE1:
            # After first shape, need "is" or "are"
            allowed = {self.is_id, self.are_id}

        elif s == State.REL_VERB:
            # After verb, need relation start: left/right/on/in
            allowed = {self.left_id, self.right_id, self.on_id, self.in_id}

        elif s == State.REL_TYPE:
            # Depends on which relation type
            if state.in_relation_of:
                # "left" or "right" -> need "of"
                allowed = {self.of_id}
            elif state.in_relation_front:
                # "in" -> need "front"
                allowed = {self.front_id}
            else:
                # "on" -> need "the"
                allowed = {self.the_id}

        elif s == State.REL_OF:
            # After "of" (from left/right of), need "the"
            allowed = {self.the_id}

        elif s == State.REL_FRONT:
            # After "front", need "of"
            allowed = {self.of_id}

        elif s == State.REL_OF2:
            # After "of" (from "in front of"), need "the"
            allowed = {self.the_id}

        elif s == State.THE2:
            # After second "the", need a color
            allowed = self.color_ids

        elif s == State.REL_COLOR2:
            # After second color, need a shape
            allowed = self.shape_ids

        elif s == State.REL_SHAPE2:
            # After second shape, need period
            allowed = {self.period_id}

        elif s == State.PERIOD:
            # After period, only <EOS> allowed
            allowed = {self.eos_id}

        elif s == State.END:
            # Terminal state
            allowed = set()

        return sorted(list(allowed))

    def advance(self, state: DecodingState, token_id: int) -> DecodingState:
        """Advance FSM state given the next token.

        Args:
            state: Current decoding state
            token_id: Next token ID

        Returns:
            New decoding state

        Raises:
            ValueError: If token_id is not valid for current state
        """
        allowed = self.allowed_ids(state)
        if token_id not in allowed:
            token_str = self.vocab.get_token(token_id)
            allowed_tokens = [self.vocab.get_token(tid) for tid in allowed]
            raise ValueError(
                f"Invalid token '{token_str}' (id={token_id}) for state {state.fsm_state}. "
                f"Allowed tokens: {allowed_tokens}"
            )

        # Determine next state based on current state and token
        s = state.fsm_state
        new_state = DecodingState(fsm_state=s, count_value=state.count_value,
                                   in_relation_of=state.in_relation_of,
                                   in_relation_front=state.in_relation_front)

        if s == State.START:
            if token_id == self.bos_id:
                new_state.fsm_state = State.START  # Stay at START
            elif token_id == self.there_id:
                new_state.fsm_state = State.THERE
            elif token_id == self.the_cap_id:  # Capitalized "The"
                new_state.fsm_state = State.THE1

        elif s == State.THERE:
            # is/are -> COUNT_VERB
            new_state.fsm_state = State.COUNT_VERB

        elif s == State.COUNT_VERB:
            # number -> COUNT_NUMBER
            # Store the count value for later plural checking
            token_str = self.vocab.get_token(token_id)
            new_state.count_value = self.number_to_count.get(token_str)
            new_state.fsm_state = State.COUNT_NUMBER

        elif s == State.COUNT_NUMBER:
            # color -> COUNT_COLOR
            new_state.fsm_state = State.COUNT_COLOR

        elif s == State.COUNT_COLOR:
            # shape -> COUNT_SHAPE
            new_state.fsm_state = State.COUNT_SHAPE

        elif s == State.COUNT_SHAPE:
            # period -> PERIOD
            new_state.fsm_state = State.PERIOD

        elif s == State.THE1:
            # color -> REL_COLOR1
            new_state.fsm_state = State.REL_COLOR1

        elif s == State.REL_COLOR1:
            # shape -> REL_SHAPE1
            new_state.fsm_state = State.REL_SHAPE1

        elif s == State.REL_SHAPE1:
            # is/are -> REL_VERB
            new_state.fsm_state = State.REL_VERB

        elif s == State.REL_VERB:
            # Relation type
            if token_id == self.left_id or token_id == self.right_id:
                new_state.in_relation_of = True
                new_state.fsm_state = State.REL_TYPE
            elif token_id == self.in_id:
                new_state.in_relation_front = True
                new_state.fsm_state = State.REL_TYPE
            elif token_id == self.on_id:
                new_state.fsm_state = State.REL_TYPE

        elif s == State.REL_TYPE:
            if state.in_relation_of and token_id == self.of_id:
                new_state.fsm_state = State.REL_OF
            elif state.in_relation_front and token_id == self.front_id:
                new_state.fsm_state = State.REL_FRONT
            elif token_id == self.the_id:
                # "on" case goes directly to THE2
                new_state.fsm_state = State.THE2

        elif s == State.REL_OF:
            # "of" after left/right -> THE2
            new_state.fsm_state = State.THE2

        elif s == State.REL_FRONT:
            # "front" -> need "of"
            new_state.fsm_state = State.REL_OF2

        elif s == State.REL_OF2:
            # "of" after "in front" -> THE2
            new_state.fsm_state = State.THE2

        elif s == State.THE2:
            # color -> REL_COLOR2
            new_state.fsm_state = State.REL_COLOR2

        elif s == State.REL_COLOR2:
            # shape -> REL_SHAPE2
            new_state.fsm_state = State.REL_SHAPE2

        elif s == State.REL_SHAPE2:
            # period -> PERIOD
            new_state.fsm_state = State.PERIOD

        elif s == State.PERIOD:
            # <EOS> -> END
            new_state.fsm_state = State.END

        return new_state

    def is_terminal(self, state: DecodingState) -> bool:
        """Check if state is terminal (decoding complete).

        Args:
            state: Current decoding state

        Returns:
            True if state is terminal
        """
        return state.fsm_state == State.END
