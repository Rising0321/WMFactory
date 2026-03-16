from abc import ABC
from typing import Any

import torch as th
import pytorch_lightning as pl

# Using relative import so that this module is easier to move from this place to elsewhere
from .tensor_spaces import TensorSpace


class EncoderDecoderBase(pl.LightningModule):
    """
    Base class for all encoders and decoders.

    Encoders turn datapoints from "world_space" to "encoder_space".
    Decoders turn datapoints from "encoder_space" to "world_space".

    All tensors are in format (batch, time, ...), where 'batch' and 'time' dimensions
    are always present (even if they are 1). Both world and encoder spaces can have
    any number of dimensions in the '...' part.
    """

    # This is a dictionary of keyword arguments that can be used to create this class
    # during testing/quick debugging (e.g., minimal model size)
    __DEBUG_CREATION_KWARGS__: dict[str, Any] = dict()

    def __init__(self):
        super().__init__()
        self._world_space = None
        self._encoder_space = None

    @property
    def world_space(self) -> TensorSpace:
        assert self._world_space is not None, "'world_space' is not defined. Set it with 'self.world_space = [TensorSpace]'."
        return self._world_space

    @world_space.setter
    def world_space(self, value: TensorSpace) -> None:
        assert isinstance(value, TensorSpace), f"'world_space' must be of type TensorSpace, but is {type(value)}"
        self._world_space = value

    @property
    def encoder_space(self) -> TensorSpace:
        assert self._encoder_space is not None, "'encoder_space' is not defined. Set it with 'self.encoder_space = [TensorSpace]'."
        return self._encoder_space

    @encoder_space.setter
    def encoder_space(self, value: TensorSpace) -> None:
        assert isinstance(value, TensorSpace), f"'encoder_space' must be of type TensorSpace, but is {type(value)}"
        self._encoder_space = value

    def encode(self, world_space_tensor: th.Tensor) -> th.Tensor:
        """
        Encodes a tensor from world space to encoder space.

        The input tensor should match the world space of this encoder.
        The input tensor may have any number of preceding dimensions (batch, time, ...),
        and output result will be parallelly encoded for the preceding dimensions.

        Args:
            world_space_tensor: Pytorch Tensor in world space (self.world_space.contains(world_space_tensor) == True)s
        Returns:
            Pytorch Tensor in encoder space (self.encoder_space.contains(return_value) == True)
        """
        if not self.world_space.contains(world_space_tensor):
            raise ValueError(f"Input tensor to `encode` {world_space_tensor} is not in world space {self.world_space}")

        preceding_dims = self.world_space.get_preceding_dimensions(world_space_tensor)
        encoder_space_tensor = self._encode(world_space_tensor)

        if not self.encoder_space.contains(encoder_space_tensor):
            raise ValueError(f"Output tensor from `_encode` {encoder_space_tensor} is not in encoder space {self.encoder_space}")

        new_preceding_dims = self.encoder_space.get_preceding_dimensions(encoder_space_tensor)
        if new_preceding_dims != preceding_dims:
            raise ValueError(f"Output tensor from `_encode` has preceding dimensions {new_preceding_dims}, but input tensor had preceding dimensions {preceding_dims}")

        return encoder_space_tensor

    def decode(self, encoder_space_tensor: th.Tensor) -> th.Tensor:
        """
        Decodes a tensor from encoder space to world space.

        The input tensor should match the encoder space of this decoder.
        The input tensor may have any number of preceding dimensions (batch, time, ...),
        and output result will be parallelly decoded for the preceding dimensions.

        Args:
            encoder_space_tensor: Pytorch Tensor in encoder space (self.encoder_space.contains(encoder_space_tensor) == True)
        Returns:
            Pytorch Tensor in world space (self.world_space.contains(return_value) == True)
        """
        if not self.encoder_space.contains(encoder_space_tensor):
            raise ValueError(f"Input tensor to `decode` {encoder_space_tensor} is not in encoder space {self.encoder_space}")

        preceding_dims = self.encoder_space.get_preceding_dimensions(encoder_space_tensor)
        world_space_tensor = self._decode(encoder_space_tensor)

        if not self.world_space.contains(world_space_tensor):
            raise ValueError(f"Output tensor from `_decode` {world_space_tensor} is not in world space {self.world_space}")

        # Make sure that the output tensor has the same preceding dimensions as the input tensor
        new_preceding_dims = self.world_space.get_preceding_dimensions(world_space_tensor)
        if new_preceding_dims != preceding_dims:
            raise ValueError(f"Output tensor from `_decode` has preceding dimensions {new_preceding_dims}, but input tensor had preceding dimensions {preceding_dims}")

        return world_space_tensor

    def _encode(self, world_space_tensor: th.Tensor) -> th.Tensor:
        raise NotImplementedError("Encoder function `_encode` not implemented")

    def _decode(self, encoder_space_tensor: th.Tensor) -> th.Tensor:
        raise NotImplementedError("Decoder function `_decode` not implemented")
