import torch as th
from tensordict import TensorDict

from wham.models.nn.nanoGPT import GPT, GPTConfig
from wham.models.wham_base.predictor import PredictorBase
from wham.models.wham_base.tensor_spaces import TensorSpace

# These are from the Chincilla paper
# https://arxiv.org/abs/2203.15556
GPT_MODEL_SIZES = {
    "chinchilla-251M": {"n_layer": 16, "n_head": 16, "n_embd": 1024},
    "chinchilla-1018M": {"n_layer": 23, "n_head": 14, "n_embd": 1792},
    "wham-4kvocab-1.6b": {"n_layer": 24, "n_head": 18, "n_embd": 2304}, # Bad name, vocab size is actually 16K
    "wham-3b": {"n_layer": 30, "n_head": 24, "n_embd": 3072},
}


def interleave_seq_token_tensors(tensors):
    """
    Interleaves tokens from different sequences.
    For example, if we have tensors (states, actions), we want to interleave them
    into a single sequence of tokens in style [state1, action1, state2, action2, ...]

    Inputs:
        tensors: list of torch tensors of shape (batch_size, seq_len, num_tokens)
                 `num_tokens` can vary between tensors.
    Outputs:
        interleaved: tensor of shape (batch_size, seq_len * (num_tokens1 + num_tokens2 + ...))
    """
    assert all(tensor.ndim == 3 for tensor in tensors), "All tensors must be 3D"
    interleaved = th.cat(tensors, dim=-1)
    interleaved = interleaved.reshape(interleaved.shape[0], -1)
    # Continuity is required for efficient memory access (and nn operations)
    return interleaved.contiguous()


def deinterleave_seq_token_tensors(interleaved, num_tokens_per_tensor):
    """
    Inverse of interleave_seq_token_tensors.
    Takes in interleaved tensor of tokens (batch_size, seq_len * (num_tokens1 + num_tokens2 + ...)),
    and returns a list of tensors of shape (batch_size, seq_len, num_tokens),
    where `num_tokens` is specified by num_tokens_per_tensor (a list of integers).

    Inputs:
        interleaved: tensor of shape (batch_size, seq_len * (num_tokens_per_tensor[0] + num_tokens_per_tensor[1] + ...))
        num_tokens_per_tensor: list of integers specifying the number of tokens per tensor
    Outputs:
        tensors: list of torch tensors of shape (batch_size, seq_len, num_tokens)
    """
    assert interleaved.ndim == 2, "Interleaved tensor must be 2D"
    num_tokens_per_step = sum(num_tokens_per_tensor)
    num_tokens = interleaved.shape[-1]

    assert num_tokens % num_tokens_per_step == 0, "Interleaved tensor must be divisible by num_tokens_per_step"
    seq_len = num_tokens // num_tokens_per_step

    matrix_interleaved = interleaved.reshape(-1, seq_len, num_tokens_per_step)
    tensors = []
    start = 0
    for num_tokens in num_tokens_per_tensor:
        tensors.append(matrix_interleaved[:, :, start : start + num_tokens])
        start += num_tokens
    return tensors


def interleave_seq_token_embedding_tensors(tensors):
    """
    Same as interleave_seq_token_tensors, but 4D tensors (instead of tokens, we have embedding vectors).

    Interleaves token embedding tensors (batch, seq_len, ?, embedding_dim) from different tensors.
    Dimension ? is the number of tokens each item in different tensors have.

    This is same as interleave_seq_token_tensors, but for token embedding tensors (additional dimension).

    For example, if we have tensors (states, actions) in following shapes:
        states: (batch_size, seq_len, tokens_per_state, embedding_dim), and
        actions: (batch_size, seq_len, tokens_per_action, embedding_dim),
    where each item (last two dimensions) represents as single state or action in multiple tokens.
    This function interleaves them into a single sequence of tokens in order
        [state1, action1, state2, action2, ...],
    with the shape
        (batch_size, seq_len * (tokens_per_state + tokens_per_action), embedding_dim)

    Inputs:
        tensors: list of torch tensors of shape (batch_size, seq_len, num_tokens, embedding_dim)
                 `num_tokens` can vary between tensors.
    Outputs:
        interleaved: tensor of shape (batch_size, seq_len * (num_tokens1 + num_tokens2 + ...), embedding_dim)
    """
    assert all(tensor.ndim == 4 for tensor in tensors), "All tensors must be 4D"
    interleaved = th.cat(tensors, dim=2)
    embedding_dim = interleaved.shape[-1]
    interleaved = interleaved.reshape(interleaved.shape[0], -1, embedding_dim)
    # Continuity is required for efficient memory access (and nn operations)
    return interleaved.contiguous()


def create_nano_gpt_model(model_size, max_context_length_tokens, vocab_size=1, version=1, bias=True):
    assert model_size in GPT_MODEL_SIZES, "Invalid model size"
    gpt_config = GPTConfig()
    gpt_config.vocab_size = vocab_size
    gpt_model_size_conf = GPT_MODEL_SIZES[model_size]
    gpt_config.n_layer = gpt_model_size_conf["n_layer"]
    gpt_config.n_head = gpt_model_size_conf["n_head"]
    gpt_config.n_embd = gpt_model_size_conf["n_embd"]
    gpt_config.block_size = max_context_length_tokens
    gpt_config.version = version
    gpt_config.bias = bias
    gpt_model = GPT(gpt_config)

    return gpt_model, gpt_config


class GPTTokenPredictor(PredictorBase):
    """
    Modality predictor that works on token basis and predicts next tokens.
    - Uses positional encoding per token
    - Uses fixed ordering of modalities
    - Autoregressive prediction
    - Tokens for each modality are interleaved into a single sequence
    - Each modality has its own set of tokens (i.e., no overlapping token indeces between modalities)
    """

    __DEBUG_CREATION_KWARGS__ = {
        "model_spec": {
            "model_size": "debug_small_width",
            "seq_len": 4,
        }
    }

    _acceptable_context_spaces = (TensorSpace((None,), dtype=th.long),)
    _acceptable_condition_spaces = tuple()

    def __init__(self, context_space, condition_space, model_spec):
        super().__init__(context_space, condition_space)
        self.save_hyperparameters()

        self.model_size = model_spec["model_size"]
        self.seq_len = model_spec["seq_len"]

        # Number of tokens per modality, as a dict and as a list (in order for (de)interleaving))
        self.tokens_per_modality = {}
        self.tokens_per_modality_list = []

        # Each modality will have its own set of tokens.
        # But also respect the individual separation of token ranges per modality.
        # Modality name -> integer, telling how much we need to offset tokens for this modality
        self.vocab_offset_per_modality = {}
        # modality name -> Tuple[int, int], telling the range of tokens for this modality, for each token individually
        self.vocab_range_per_modality = {}
        # List of modality names, in the order they are interleaved into the sequence.
        self.modality_order = []
        self.total_vocab_size = 0
        for name, space in self.context_space.items():
            assert space.high is not None and space.low is not None, "High and low must be specified for all context spaces (vocab size per modality)"
            self.tokens_per_modality[name] = space.shape[0]
            self.tokens_per_modality_list.append(space.shape[0])

            low_tensor = space.low
            high_tensor = space.high

            min_token = low_tensor.min().item()
            assert min_token == 0, f"Lowest token of space {name} is {min_token}, but must be 0. This is for clarity and avoiding unused tokens"
            max_token = high_tensor.max().item()

            self.vocab_offset_per_modality[name] = self.total_vocab_size
            self.vocab_range_per_modality[name] = tuple(
                (int(low.item()) + self.total_vocab_size, int(high.item()) + self.total_vocab_size) for low, high in zip(low_tensor, high_tensor)
            )
            # +1 because high is inclusive, and we ensured that low is 0
            self.total_vocab_size += max_token + 1

            self.modality_order.append(name)

        # Adjust the tokens allowed when generating the very first image token
        # All we are doing is disallowing the 0th token
        # NOTE: THIS IS *NOT* required at all
        # However, it provides a better experience when using the 200M model trained on 128x128 images
        # This is a bit fiddly since tuples don't support item assignment
        list_version_of_tuple = list(self.vocab_range_per_modality["images"])
        list_version_of_tuple[0] = (list_version_of_tuple[0][0] + 1, list_version_of_tuple[0][1])
        self.vocab_range_per_modality["images"] = tuple(list_version_of_tuple)
        # End of adjustment for the first image token

        self.total_tokens_per_step = sum(self.tokens_per_modality.values())
        self.seq_len_in_tokens = self.seq_len * self.total_tokens_per_step
        self.seq_len_in_tokens_for_inference = (self.seq_len - 1) * self.total_tokens_per_step

        gpt_version = model_spec.get("nanogpt_version", 1)
        gpt_bias = model_spec.get("bias", True)
        print(f"Creating NanoGPT model with version {gpt_version}")
        self.gpt_model, self.gpt_config = create_nano_gpt_model(self.model_size, vocab_size=self.total_vocab_size, max_context_length_tokens=self.seq_len_in_tokens, version=gpt_version, bias=gpt_bias)

    def parameters(self):
        return self.gpt_model.parameters()

    def _create_gpt(self, model_size, vocab_size, max_context_length_tokens):
        assert model_size in GPT_MODEL_SIZES, "Invalid model size"
        gpt_config = GPTConfig()
        gpt_config.vocab_size = vocab_size
        gpt_model_size_conf = GPT_MODEL_SIZES[model_size]
        gpt_config.n_layer = gpt_model_size_conf["n_layer"]
        gpt_config.n_head = gpt_model_size_conf["n_head"]
        gpt_config.n_embd = gpt_model_size_conf["n_embd"]
        gpt_config.block_size = max_context_length_tokens
        gpt_model = GPT(gpt_config)

        return gpt_model, gpt_config

    def _check_not_too_long_context_length(self, tokens, num_tokens_to_be_generated):
        """
        Check that the context length is not too long for the amount we are trying to generate
        """
        if (tokens.shape[1] + num_tokens_to_be_generated) > self.seq_len_in_tokens:
            raise ValueError(
                f"Trying to generate too many tokens given the context. Context {tokens.shape[1]} should be less than {self.seq_len_in_tokens} - {num_tokens_to_be_generated}"
            )

    def _interleave_and_offset_modalities(self, modalities):
        """
        Interleave and offset tokens from different modalities into a single sequence.
        Offset tokens of each modality so that different modalities do not overlap.

        Assumes modalities is already checked to be valid input for the model.
        """
        modality_list = [modalities[name] + self.vocab_offset_per_modality[name] for name in self.modality_order]
        interleaved_tokens = interleave_seq_token_tensors(modality_list)
        return interleaved_tokens

    def _deinterleave_and_offset_tokens(self, interleaved_tokens):
        """
        Inverse of _interleave_and_offset_modalities
        """
        modality_list = deinterleave_seq_token_tensors(interleaved_tokens, self.tokens_per_modality)
        modality_list = {name: modality_list[i] - self.vocab_offset_per_modality[name] for i, name in enumerate(self.modality_order)}
        return modality_list

    def predict_n_tokens(self, tokens, n_tokens, valid_token_ranges, deterministic=False, temperature=1.0, top_k=None, top_p=None, min_tokens_to_keep=1):
        """
        Given a sequence of tokens, predict the next action.
        Returns new list of tokens with the predicted action appended.

        Inputs:
            tokens: torch tensor (batch_size, seq_len)
            n_tokens: int, number of tokens to predict
            valid_token_ranges: Tuple[int, int] of valid vocab indices to predict from for each token (inclusive on both sides)
            **kwargs: kwargs for gpt_model.optimized_generate
        """
        self._check_not_too_long_context_length(tokens, n_tokens)
        assert n_tokens == len(
            valid_token_ranges
        ), f"Must have a valid token range for each token to be generated. Expected {n_tokens}, got valid_token_ranges of length {len(valid_token_ranges)}"
        new_tokens = self.gpt_model.optimized_generate(
            tokens,
            n_tokens,
            valid_token_ranges=valid_token_ranges,
            raise_cropping=True,
            deterministic=deterministic,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_tokens_to_keep=min_tokens_to_keep,
        )
        return new_tokens

    def cross_entropy_prediction_loss_on_tokens(self, token_seq, loss_mask):
        """
        Given a sequence of tokens, try to predict next tokens on every timestep given all previous timesteps.
        Returns average loss.

        Inputs:
            token_seq: torch tensor (batch_size, token_seq_len)
                       dtype: th.long (0 <= token < vocab_size)
            mask: torch tensor (batch_size, token_seq_len). 1 if timestep is valid, 0 if timestep is padding
        Outputs:
            losses: loss per timestep (batch_size, token_seq_len), where first timestep loss is 0 (padded)
        """
        inputs = token_seq[:, :-1].contiguous()
        targets = token_seq[:, 1:].contiguous()
        loss_mask = loss_mask[:, 1:].contiguous()
        _, losses = self.gpt_model(inputs, targets=targets, loss_mask=loss_mask, loss_reduction="none")
        # Pad from the left with zeros (there is no valid target for the first step)
        losses = th.cat([th.zeros_like(losses[:, :1]), losses], dim=1)
        return losses

    def cross_entropy_prediction_loss(self, modalities, loss_mask):
        """
        Given a TensorDict of sequence of different modalities as tokens, interleave them into a single sequence
        and try to predict next tokens on every timestep given all previous timesteps.
        Returns average loss.

        Inputs:
            modalities: TensorDict of modality name -> (batch_size, seq_len, tokens)
            loss_mask: torch tensor (batch_size, seq_len). 1 if timestep is valid, 0 if timestep is padding
        Outputs:
            modality_losses: dictionary of losses per modality (batch_size, seq_len), where first timestep loss is 0 (padded)
            n_valid_tokens: number of valid tokens in the loss mask
        """
        self.assert_check_context_tensordict_is_valid(modalities)
        interleaved_tokens = self._interleave_and_offset_modalities(modalities)
        # Mask is just repeated the same number of times as there are tokens per timestep
        loss_mask = loss_mask.repeat_interleave(dim=1, repeats=self.total_tokens_per_step)
        losses = self.cross_entropy_prediction_loss_on_tokens(interleaved_tokens, loss_mask)

        # Split loss into different modalities
        split_losses = deinterleave_seq_token_tensors(losses, self.tokens_per_modality_list)

        modality_losses = {name: loss for name, loss in zip(self.modality_order, split_losses)}

        # Compute average loss.
        # To match with the previous numbers, where whole loss was summed over all tokens,
        # we divide by the number of all valid tokens, not just the number of timesteps.
        num_valid_tokens = loss_mask.sum()
        return modality_losses, num_valid_tokens

    def predict_next_step(self, modalities, modalities_to_predict=None, **kwargs):
        """
        Given a TensorDict of sequence of different modalities as tokens, predict the tokens for the next step.

        Inputs:
            modalities: TensorDict of modality name -> (batch_size, seq_len, tokens)
            modalities_to_predict: list of modalities to predict. If None, predict all modalities
                                   NOTE: modalities_to_predict must be a subset of self.modality_order and in the same order.
                                         e.g., if model was trained to predict steps in order [image, action], you can not predict
                                         "action" first, as the model requires the image tokens first.
            **kwargs: kwargs for gpt_model.optimized_generate
        Outputs:
            predicted_modalities: TensorDict of predicted modalities
            all_tokens: tensor (batch_size, seq_len, tokens) of all tokens, including predicted ones
        """
        self.assert_check_context_tensordict_is_valid(modalities)
        # We have to manually avoid cutting down on context, as otherwise inference would fail (first token in context
        # has to _always_ be first token of an image).
        if modalities.shape[1] == self.seq_len:
            modalities = modalities[:, 1:]

        all_tokens = self._interleave_and_offset_modalities(modalities)
        predicted_tokens = dict()

        modalities_to_predict = modalities_to_predict or self.modality_order
        for desired_modality, modality_name in zip(modalities_to_predict, self.modality_order):
            assert (
                desired_modality == modality_name
            ), f"Modalities to predict {modalities_to_predict} was in wrong order. Must follow the ordering of {self.modality_order}"

        for modality_name in self.modality_order:
            tokens_to_predict = self.tokens_per_modality[modality_name]
            all_tokens = self.predict_n_tokens(all_tokens, tokens_to_predict, self.vocab_range_per_modality[modality_name], **kwargs)
            predicted_tokens[modality_name] = all_tokens[:, -tokens_to_predict:] - self.vocab_offset_per_modality[modality_name]
            # Add time dimension
            predicted_tokens[modality_name] = predicted_tokens[modality_name].unsqueeze(1)

        batch_dimension = self.context_space.get_preceding_dimensions(modalities)[0]
        predicted_modalities = TensorDict(predicted_tokens, batch_size=(batch_dimension, 1))
        return predicted_modalities, tokens_to_predict
