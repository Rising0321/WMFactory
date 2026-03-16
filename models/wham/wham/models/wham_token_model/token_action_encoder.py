import torch as th

from wham.models.wham_base.tensor_spaces import TensorSpace
from wham.models.wham_base.encoder_decoder import EncoderDecoderBase

# Each binary button will get on/off token associated with that button.
# This is to replicate original world model experiments.
# Potentially, could also try having only one on and one off token which
# is used by all buttons.
VOCAB_SIZE_FOR_BUTTON = 2

MAX_BUTTONS = 12
POS_ACTIONS = 4
POS_CLASSES = 11


def get_valid_token_range_for_action_idx(action_idx, n_bins, token_offset=0):
    """
    Given index of action token, return the range of valid token indices
    for that action index, inclusive on both sides

    Inputs:
        action_idx: index of action token
        n_bins: number of bins used for stick discretization
        token_offset: offset to add to token indices
    Outputs:
        valid_token_range: (min_token_idx, max_token_idx)
    """
    if action_idx < MAX_BUTTONS:
        # Button
        min_token_idx = action_idx * VOCAB_SIZE_FOR_BUTTON
        max_token_idx = min_token_idx + VOCAB_SIZE_FOR_BUTTON - 1
    else:
        # Stick
        min_token_idx = MAX_BUTTONS * VOCAB_SIZE_FOR_BUTTON + (action_idx - MAX_BUTTONS) * n_bins
        max_token_idx = min_token_idx + n_bins - 1
    return min_token_idx + token_offset, max_token_idx + token_offset


def tokenize_actions(action_seq_batch, n_bins, token_offset=0):
    """
    Tokenize BE actions into a sequence of tokens:
        - Buttons are mapped to on/off tokens. Each button has its unique on/off tokens.
        - Stick actions (which should be discrete) are mapped to unique tokens per stick.

    Inputs:
        action_seq_batch: torch tensor (batch_size, seq_len, MAX_BUTTONS + POS_ACTIONS)
        n_bins: number of bins used for stick discretization
        token_offset: offset to add to token indices to avoid overlap with state tokens
    Outputs:
        action_seq_batch_discrete: (batch_size, seq_len, MAX_BUTTONS + POS_ACTIONS)
    """
    # Make sure we get what we expect
    assert action_seq_batch.shape[-1] == MAX_BUTTONS + POS_ACTIONS
    action_token_seq_batch = th.zeros_like(action_seq_batch).long()

    # Buttons
    total_token_offset = token_offset
    for button_i in range(MAX_BUTTONS):
        # Unique on/off token for every button
        action_token_seq_batch[:, :, button_i] = (action_seq_batch[:, :, button_i] + button_i * VOCAB_SIZE_FOR_BUTTON + total_token_offset).long()

    total_token_offset += MAX_BUTTONS * VOCAB_SIZE_FOR_BUTTON
    for action_index in range(MAX_BUTTONS, MAX_BUTTONS + POS_ACTIONS):
        stick_index = action_index - MAX_BUTTONS
        action_token_seq_batch[:, :, action_index] = (action_seq_batch[:, :, action_index] + stick_index * n_bins + total_token_offset).long()

    return action_token_seq_batch


def detokenize_actions(action_token_seq_batch, n_bins, token_offset=0):
    """
    Reverse of tokenize_actions. See tokenize_actions for details.
    Note that this returns discretized actions for sticks, which follow the discretization scheme of
    rest of this repository (see e.g., data.parser.action_plugins)
    """
    action_seq_batch = th.zeros_like(action_token_seq_batch).float()

    # Buttons
    total_token_offset = token_offset
    for button_i in range(MAX_BUTTONS):
        action_seq_batch[:, :, button_i] = (action_token_seq_batch[:, :, button_i] - button_i * VOCAB_SIZE_FOR_BUTTON - total_token_offset).float()

    total_token_offset += MAX_BUTTONS * VOCAB_SIZE_FOR_BUTTON
    # Assume rest are continuous actions
    for action_index in range(MAX_BUTTONS, MAX_BUTTONS + POS_ACTIONS):
        stick_index = action_index - MAX_BUTTONS
        action_bin = action_token_seq_batch[:, :, action_index] - stick_index * n_bins - total_token_offset
        action_seq_batch[:, :, action_index] = action_bin
    return action_seq_batch


def get_action_vocab_size(bins_for_sticks):
    """Return vocab size required by buttons"""
    # Each button has 2 tokens (on/off), each stick has n_bins_for_sticks tokens (unique to every button/stick)
    return MAX_BUTTONS * VOCAB_SIZE_FOR_BUTTON + POS_ACTIONS * bins_for_sticks


class ActionTokenEncoder(EncoderDecoderBase):
    """
    Encoder for turning BE actions into sequence of tokens
    """

    __DEBUG_CREATION_KWARGS__ = dict()

    def __init__(self):
        super().__init__()
        self.n_bins_for_sticks = POS_CLASSES
        self.vocab_size = get_action_vocab_size(self.n_bins_for_sticks)

        action_dim = MAX_BUTTONS + POS_ACTIONS

        # Original actions have buttons {0, 1} and then discretized positions [0, POS_CLASSES - 1]
        world_space_lows = th.tensor([0] * MAX_BUTTONS + [0] * POS_ACTIONS, dtype=th.float)
        world_space_highs = th.tensor([1] * MAX_BUTTONS + [POS_CLASSES - 1] * POS_ACTIONS, dtype=th.float)
        self.world_space = TensorSpace((action_dim,), dtype=th.float, low=world_space_lows, high=world_space_highs)

        # In encoder space, each button has its own on/off token, and each stick has n_bins_for_sticks tokens
        self._action_token_ranges = [get_valid_token_range_for_action_idx(i, self.n_bins_for_sticks) for i in range(action_dim)]
        encoder_space_lows = th.tensor([r[0] for r in self._action_token_ranges], dtype=th.long)
        encoder_space_highs = th.tensor([r[1] for r in self._action_token_ranges], dtype=th.long)
        self.encoder_space = TensorSpace((action_dim,), dtype=th.long, low=encoder_space_lows, high=encoder_space_highs)

    def _encode(self, world_space_tensor):
        """
        Encode BE actions into tokens
        """
        assert world_space_tensor.ndim == 3, "ActionTokenEncoder only supports (batch, seq_len, action_dim) tensors"
        return tokenize_actions(world_space_tensor, self.n_bins_for_sticks)

    def _decode(self, encoder_space_tensor):
        """
        Decode tokens into BE actions
        """
        assert encoder_space_tensor.ndim == 3, "ActionTokenEncoder only supports (batch, seq_len, action_dim) tensors"
        return detokenize_actions(encoder_space_tensor, self.n_bins_for_sticks)
