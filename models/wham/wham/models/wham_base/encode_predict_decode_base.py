from typing import Any, Union, Type, Callable, Tuple, Mapping, Optional

import torch as th
import pytorch_lightning as pl
from tensordict import TensorDict  # type: ignore # requires installing stubs for tensordict

from .tensor_spaces import TensorDictSpace
from .encoder_decoder import EncoderDecoderBase
from .pl_creation_args import LightningModuleCreationArgs


def create_encoder_args_from_config_dict(
    config_dict: dict[str, Union[dict[str, Any], tuple]], class_name_to_model: Callable[[str], Type[pl.LightningModule]]
) -> Mapping[str, Union[LightningModuleCreationArgs, Tuple[LightningModuleCreationArgs, LightningModuleCreationArgs]]]:
    """
    Given a dictionary mapping modality names to their encoder-decoder arguments, create the corresponding
    creation args (LightningModuleCreationArgs) for each modality.

    See LightningModuleCreationArgs.from_dict for more details.

    Args:
        config_dict: A dictionary mapping modality names to their encoder-decoder arguments.
                     Root level of this dictionary should be modality names we expect.
        class_name_to_model: A function mapping class names to their corresponding model classes.

    Returns:
        A dictionary mapping modality names to their encoder-decoder creation args.
        Each value may be a LightningModuleCreationArgs, or a tuple of two LightningModuleCreationArgs.
        If value is a LightningModuleCreationArgs, then same model is used for encoding and decoding.
        If value is a tuple of two LightningModuleCreationArgs, then first is used for encoding and second for decoding.
    """
    # Giving explicit type hint here to make mypy happy
    modalities: dict[str, Any] = {}
    for modality_name, modality_config in config_dict.items():
        if isinstance(modality_config, (list, tuple)):
            assert len(modality_config) == 2, f"Expected two entries for modality {modality_name}, got {len(modality_config)}"
            modalities[modality_name] = (
                LightningModuleCreationArgs.from_dict(modality_config[0], class_name_to_model),
                LightningModuleCreationArgs.from_dict(modality_config[1], class_name_to_model),
            )
        else:
            modalities[modality_name] = LightningModuleCreationArgs.from_dict(modality_config, class_name_to_model)
    return modalities


def create_encoder_modules_from_args(
    encoders: Mapping[str, Union[LightningModuleCreationArgs, Tuple[LightningModuleCreationArgs, LightningModuleCreationArgs]]], remove_checkpoint_path: bool = True
) -> th.nn.ModuleDict:
    """
    Create the encoder modules from given creation args (LightningModuleCreationArgs).

    Args:
        encoders: A dictionary mapping modality names to their encoder-decoder creation args.
                  If value is a LightningModuleCreationArgs, then same model is used for encoding and decoding.
                  If value is a tuple of two LightningModuleCreationArgs, then first is used for encoding and second for decoding.
        remove_checkpoint_path: If True, then remove the checkpoint_path from the creation args. This prepares the
                                created moduled to be properly saved and loaded as part of the bigger model

    Returns:
        A dictionary mapping modality names to their encoder-decoder modules.
    """
    modalities = {}
    for modality_name, modality_args in encoders.items():
        if isinstance(modality_args, (list, tuple)):
            modalities[modality_name] = th.nn.ModuleList(
                [
                    modality_args[0].create_module(remove_checkpoint_path=remove_checkpoint_path),
                    modality_args[1].create_module(remove_checkpoint_path=remove_checkpoint_path),
                ]
            )
        else:
            modalities[modality_name] = modality_args.create_module(remove_checkpoint_path=remove_checkpoint_path)
    return th.nn.ModuleDict(modalities)


class EncodePredictDecodeModule(pl.LightningModule):
    """
    Base-class for models that encode, predict and decode.

    Args:
        context_encoders: A dictionary mapping modality names to their encoder-decoders.
                          If value is a pl.LightningModule, then same model is used for encoding and decoding.
                          If value is a tuple of two pl.LightningModule, then first is used for encoding and second for decoding.
        condition_encoders: Same as `context_encoders`, but for conditions.
    """

    def __init__(
        self,
        predictor_args: LightningModuleCreationArgs,
        context_encoders: th.nn.ModuleDict,
        condition_encoders: Optional[th.nn.ModuleDict] = None,
    ):
        if condition_encoders is None:
            condition_encoders = th.nn.ModuleDict(dict())
        self._assert_encoders(context_encoders)
        self._assert_encoders(condition_encoders)
        super().__init__()

        self.context_encoders = context_encoders
        self.condition_encoders = condition_encoders

        self.context_world_space, self.context_encoder_space = self._get_spaces_from_encoders(context_encoders)
        self.condition_world_space, self.condition_encoder_space = self._get_spaces_from_encoders(condition_encoders)

        self.predictor = predictor_args.create_module(context_space=self.context_encoder_space, condition_space=self.condition_encoder_space)

    def _assert_encoders(self, encoders: th.nn.ModuleDict) -> None:
        """Check that encoder dictionary is valid"""
        assert isinstance(encoders, th.nn.ModuleDict), f"Invalid type for encoders: {type(encoders)}. Expected th.nn.ModuleDict"
        for modality_name, encoder in encoders.items():
            assert isinstance(encoder, EncoderDecoderBase) or isinstance(
                encoder, th.nn.ModuleList
            ), f"Invalid type for modality {modality_name}: {type(encoder)}. Expected EncoderDecoderBase or Tuple[EncoderDecoderBase]"
            if isinstance(encoder, th.nn.ModuleList):
                assert len(encoder) == 2, f"Invalid number of arguments for modality {modality_name}: {len(encoder)}. Expected two (encoder, decoder)"
                assert isinstance(
                    encoder[0], EncoderDecoderBase
                ), f"Invalid type for encoder of modality {modality_name}: {type(encoder[0])}. Expected EncoderDecoderBase"
                assert isinstance(
                    encoder[1], EncoderDecoderBase
                ), f"Invalid type for decoder of modality {modality_name}: {type(encoder[1])}. Expected EncoderDecoderBase"

    def _get_spaces_from_encoders(self, encoders: th.nn.ModuleDict) -> Tuple[TensorDictSpace, TensorDictSpace]:
        """
        Given a modality dictionary mapping modality names to their encoders and decoders,
        extract the world space and encoder space,
        """
        world_spaces = {}
        encoder_spaces = {}
        for modality_name, modality in encoders.items():
            if isinstance(modality, EncoderDecoderBase):
                encoder_spaces[modality_name] = modality.encoder_space
                world_spaces[modality_name] = modality.world_space
            elif isinstance(modality, th.nn.ModuleList):
                assert len(modality) == 2, f"Invalid number of modules for modality {modality_name}: {len(modality)}. Expected 2."
                # Make sure that both encoder and decoder spaces match the expected space
                encoder_encoder_space = modality[0].encoder_space
                decoder_encoder_space = modality[1].encoder_space
                assert (
                    encoder_encoder_space == decoder_encoder_space
                ), f"Encoder and decoder spaces for modality {modality_name} do not match: {encoder_encoder_space} != {decoder_encoder_space}"
                encoder_world_space = modality[0].world_space
                decoder_world_space = modality[1].world_space
                assert (
                    encoder_world_space == decoder_world_space
                ), f"Encoder and decoder world spaces for modality {modality_name} do not match: {encoder_world_space} != {decoder_world_space}"
                encoder_spaces[modality_name] = encoder_encoder_space
                world_spaces[modality_name] = encoder_world_space
            else:
                raise TypeError(f"Invalid type for modality {modality_name}: {type(modality)}. Expected EncoderDecoderBase or th.nn.ModuleList")
        return TensorDictSpace(world_spaces), TensorDictSpace(encoder_spaces)

    def _encode(self, input_td: TensorDict, encoders: th.nn.ModuleDict, space: TensorDictSpace) -> TensorDict:
        """
        Encode input_td into encoder space using the given encoders.

        Args:
            input_td: A tensordict mapping modality names to their inputs.
            encoders: A dictionary mapping modality names to their encoders.

        Returns:
            An encoded tensordict.
        """
        encoded_context = {}
        preceding_dims = space.get_preceding_dimensions(input_td, allow_key_subset=True)
        for modality_name in input_td.keys():
            encoder = encoders[modality_name]
            if isinstance(encoder, EncoderDecoderBase):
                encoded_context[modality_name] = encoder.encode(input_td[modality_name])
            elif isinstance(encoder, th.nn.ModuleList):
                encoded_context[modality_name] = encoder[0].encode(input_td[modality_name])
            else:
                raise TypeError(f"Invalid type for modality {modality_name}: {type(encoder)}. Expected EncoderDecoderBase or th.nn.ModuleList")
        return TensorDict(encoded_context, batch_size=preceding_dims)

    def _decode(self, input_td: TensorDict, encoders: th.nn.ModuleDict, space: TensorDictSpace) -> TensorDict:
        """
        Decode input_td into the original space using the given encoders.

        Args:
            input_td: A tensordict mapping modality names to their encoded inputs.
            encoders: A dictionary mapping modality names to their encoders.

        Returns:
            A decoded tensordict.
        """
        decoded_context = {}
        preceding_dims = space.get_preceding_dimensions(input_td, allow_key_subset=True)
        for modality_name in input_td.keys():
            encoder = encoders[modality_name]
            if isinstance(encoder, EncoderDecoderBase):
                decoded_context[modality_name] = encoder.decode(input_td[modality_name])
            elif isinstance(encoder, th.nn.ModuleList):
                decoded_context[modality_name] = encoder[1].decode(input_td[modality_name])
            else:
                raise TypeError(f"Invalid type for modality {modality_name}: {type(encoder)}. Expected EncoderDecoderBase or th.nn.ModuleList")
        return TensorDict(decoded_context, batch_size=preceding_dims)

    def encode_context(self, context: TensorDict) -> TensorDict:
        """
        Encode the given context into the encoder space.

        Args:
            context: A tensordict mapping modality names to their inputs.

        Returns:
            An encoded tensordict.
        """
        assert self.context_world_space.contains(context, allow_key_subset=True), f"Context {context} is not contained in context world space {self.context_world_space}"
        return self._encode(context, self.context_encoders, self.context_world_space)

    def decode_context(self, encoded_context: TensorDict) -> TensorDict:
        """
        Decode the given encoded context into the original space.

        Args:
            encoded_context: A tensordict mapping modality names to their encoded inputs.

        Returns:
            A decoded tensordict.
        """
        assert self.context_encoder_space.contains(
            encoded_context,
            allow_key_subset=True,
        ), f"Encoded context {encoded_context} is not contained in context encoder space {self.context_encoder_space}"
        return self._decode(encoded_context, self.context_encoders, self.context_encoder_space)

    def encode_condition(self, condition: TensorDict) -> TensorDict:
        """
        Encode the given condition into the encoder space.

        Args:
            condition: A tensordict mapping modality names to their inputs.

        Returns:
            An encoded tensordict.
        """
        assert self.condition_world_space.contains(
            condition, allow_key_subset=True
        ), f"Condition {condition} is not contained in condition world space {self.condition_world_space}"
        return self._encode(condition, self.condition_encoders, self.condition_world_space)

    def decode_condition(self, encoded_condition: TensorDict) -> TensorDict:
        """
        Decode the given encoded condition into the original space.

        Args:
            encoded_condition: A tensordict mapping modality names to their encoded inputs.

        Returns:
            A decoded tensordict.
        """
        assert self.condition_encoder_space.contains(
            encoded_condition, allow_key_subset=True
        ), f"Encoded condition {encoded_condition} is not contained in condition encoder space {self.condition_encoder_space}"
        return self._decode(encoded_condition, self.condition_encoders, self.condition_encoder_space)
