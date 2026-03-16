# Akin to gym.spaces but for pytorch

from typing import Tuple, Union, Iterator, Optional

import torch as th
from tensordict import TensorDict  # type: ignore # requires installing stubs for tensordict


def assert_is_valid_space_shape_tuple(shape: Tuple[Union[int, None]]) -> None:
    """
    Check if a tuple if valid for defining a space shape.
    A space shape is a tuple of integers or "None". A "None" indicates a wildcard dimension,

    Args:
        shape: The tuple to check.
    """
    assert isinstance(shape, tuple), f"Input {shape} is not a tuple."
    assert len(shape) > 0, f"Input tuple {shape} is not a valid shape. It must have at least one dimension."
    for dim in shape:
        if dim is not None and not isinstance(dim, int):
            raise ValueError(f"Input tuple {shape} is not a valid shape. Entry '{dim}' is not an integer or None.")


def check_if_shape_is_in_space_shape(shape: Union[Tuple[Optional[int]], th.Size], space_shape: Tuple[Union[int, None]]) -> bool:
    """
    Check if a shape is inside space shape.

    A shape is inside a space shape if the number of dimensions match and all dimensions are equal.
    If space shape has a wildcard "None", then the corresponding dimension in the shape can be int or None.
    If shape has a wildcard "None", then the corresponding dimension in the space shape must be None.

    Args:
        shape: The tensor shape to check.
        space_shape: The space shape to check.

    Returns:
        True if the tensor shape is in the space shape, False otherwise.
    """
    assert len(shape) == len(space_shape), "Input shapes must have the same number of dimensions."
    for tensor_dim, space_dim in zip(shape, space_shape):
        if space_dim is None:
            continue
        if tensor_dim != space_dim:
            return False
    return True


def prepend_dimensions(tensor: th.Tensor, dimensions_to_add: tuple) -> th.Tensor:
    """
    Prepend dimensions to a tensor by repeating the tensor on the new dimensions.
    E.g., if original shape is [3, 12, 12], and dimensions_to_add is (1, 2), then the output shape is [1, 2, 3, 12, 12].

    Args:
        tensor: The tensor to prepend dimensions to.
        dimensions_to_add: tuple of dimensions to add.

    Returns:
        The tensor with prepended dimensions.
    """
    expected_shape = tuple(dimensions_to_add) + tensor.shape
    repeat_dims = list(dimensions_to_add) + [1] * len(tensor.shape)
    output = tensor.repeat(*repeat_dims)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}."
    return output


def convert_potential_scalar_to_full_tensor(tensor_or_scalar: Union[int, float, th.Tensor], shape: Tuple[Union[int, None]], dtype: th.dtype) -> th.Tensor:
    """
    Convert a potential scalar to a broadcastable tensor.
    This is used to convert "low" and "high" arguments to TensorSpace to tensors.

    If shape has Nones (wildcards), these will be replaced with dimension 1 in the output tensor.
    This will allow broadcastable operations when doing comparisons

    Args:
        tensor_or_scalar: The potential scalar to convert to a tensor.
        shape: The shape of the tensor to convert to.
        dtype: The dtype of the tensor to convert to.

    Returns:
        The converted tensor.
    """
    assert isinstance(shape, tuple), f"Input shape {shape} is not a tuple."
    assert_is_valid_space_shape_tuple(shape)
    assert isinstance(dtype, th.dtype), f"Input dtype {dtype} is not a torch dtype."

    target_shape = tuple(1 if dim is None else dim for dim in shape)
    if isinstance(tensor_or_scalar, (int, float)):
        output_tensor = th.full(target_shape, tensor_or_scalar, dtype=dtype)
    elif isinstance(tensor_or_scalar, th.Tensor):
        # Make sure the tensor is broadcastable
        assert tensor_or_scalar.shape == target_shape, f"Input tensor shape {tensor_or_scalar.shape} is not broadcastable to shape {shape}. None's should be 1's."
        assert tensor_or_scalar.dtype == dtype, f"Input tensor dtype {tensor_or_scalar.dtype} is not expected dtype {dtype}."
        output_tensor = tensor_or_scalar
    else:
        raise ValueError(f"Input {tensor_or_scalar} is not a scalar or tensor.")
    return output_tensor


class TensorSpace:
    """
    Base class for defining a space for pytorch tensors. Similar to gym.spaces, but for pytorch.
    A space can be used to define the accepted/expected shape and dtype of a tensor.

    NOTE: tensors can have any number of preceding dimensions, but the last dimensions must match.
          E.g. if space defines a 3D image of (channel, height, width), then a 4D tensor of
          (batch, channel, height, width) is accepted, but a 4D tensor of (batch, height, width, channel)
          is not accepted.

    Args:
        shape: The expected shape of the tensor. This is tuple of integers or "None". A "None" indicates a wildcard
               dimension, i.e. any integer is accepted.
        dtype: The expected dtype of the tensor. This is strictly enforced*, i.e. even if the dtype was "castable"
               to the expected dtype, it will not be accepted.
               (*) Special case: float32 space can be casted to float16 in `contains` function to allow mixed precision training.
        low: The lower bound of the tensor (inclusive, optional). Can be a scalar or torch tensor of the same shape as "shape"
        high: The upper bound of the tensor (inclusive, optional). Can be a scalar or torch tensor of the same shape as "shape"
    """

    def __init__(
        self,
        shape: Tuple[Union[int, None]],
        dtype: th.dtype = th.float,
        low: Optional[Union[int, float, th.Tensor]] = None,
        high: Optional[Union[int, float, th.Tensor]] = None,
    ):
        assert_is_valid_space_shape_tuple(shape)
        assert isinstance(dtype, th.dtype), f"Input dtype {dtype} is not a torch dtype."
        if low is not None:
            low = convert_potential_scalar_to_full_tensor(low, shape, dtype)
        if high is not None:
            high = convert_potential_scalar_to_full_tensor(high, shape, dtype)
        if low is not None and high is not None:
            assert th.all(low <= high), f"Input low {low} is not <= high {high}."

        self._shape = shape
        self._ndim = len(shape)
        self._dtype = dtype
        self._low = low
        self._high = high
        self.device = th.device("cpu")
        self.to(self.device)

    def _check_and_move_device_if_necessary(self, tensor_or_space: Union[th.Tensor, "TensorSpace"]) -> None:
        """
        Check this TensorSpace is in right device to do operations with the tensor.
        If not, move this TensorSpace to the device of the tensor.

        Args:
            tensor: The tensor to check for.
        """
        if tensor_or_space.device != self.device:
            self.to(tensor_or_space.device)

    @property
    def shape(self) -> Tuple[Union[int, None]]:
        return self._shape

    @property
    def dtype(self) -> th.dtype:
        return self._dtype

    @property
    def low(self) -> Optional[th.Tensor]:
        return self._low

    @property
    def high(self) -> Optional[th.Tensor]:
        return self._high

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, TensorSpace), f"Input {other} is not a TensorSpace."
        if self.shape != other.shape:
            return False
        if self.dtype != other.dtype:
            return False

        if isinstance(self.low, th.Tensor) and isinstance(other.low, th.Tensor):
            if not th.all(self.low == other.low):
                return False
        elif self.low != other.low:
            return False

        if isinstance(self.high, th.Tensor) and isinstance(other.high, th.Tensor):
            if not th.all(self.high == other.high):
                return False
        elif self.high != other.high:
            return False

        return True

    def to(self, device: th.device) -> "TensorSpace":
        """
        Move the space to a device.

        Args:
            device: The device to move the space to.

        Returns:
            The space on the device.
        """

        if self._low is not None:
            self._low = self._low.to(device)
        if self._high is not None:
            self._high = self._high.to(device)
        self.device = device
        return self

    def __repr__(self) -> str:
        return f"(TensorSpace shape={self.shape}, dtype={self.dtype}, low={self.low}, high={self.high})"

    def contains(self, tensor: th.Tensor) -> bool:
        """
        Check if a tensor is in the space.

        Args:
            tensor: The tensor to check.

        Returns:
            True if the tensor is in the space, False otherwise.
        """
        if tensor.ndim < self._ndim:
            return False
        # Only check the trailing dimensions
        x_shape = tensor.shape[-self._ndim :]
        if not check_if_shape_is_in_space_shape(x_shape, self._shape):
            return False

        if tensor.dtype != self.dtype:
            # Special case for mixed-precision training: allow float32 to be casted to float16
            if not (self.dtype == th.float32 and tensor.dtype == th.float16):
                return False

        self._check_and_move_device_if_necessary(tensor)
        if self._low is not None and th.any(tensor < self._low):
            return False
        if self._high is not None and th.any(tensor > self._high):
            return False

        return True

    def is_subset_of(self, other: "TensorSpace") -> bool:
        """
        Check if this space is a subset of another space.

        A subset "sub" is subset of "super" if:
        - Dtypes are the same
        - Bounds of "sub" are within bounds of "super"
        - Shapes are the same, except for wildcards

        NOTE: Space is subset of itself.

        Args:
            other: The other space to check.

        Returns:
            True if this space is a subset of the other space, False otherwise.
        """
        assert isinstance(other, TensorSpace), f"Input {other} is not a TensorSpace."

        if self.dtype != other.dtype:
            return False

        # Manually check ndim so we can return False
        if self._ndim != other._ndim:
            return False

        if not check_if_shape_is_in_space_shape(self.shape, other.shape):
            return False

        self._check_and_move_device_if_necessary(other)
        # If other (super) does not have bounds, we can skip the checks.
        # But if other has bounds, then this (sub) must have bounds as well.
        if other.low is not None:
            if self.low is None or th.any(self.low < other.low):
                return False
        if other.high is not None:
            if self.high is None or th.any(self.high > other.high):
                return False

        return True

    def sample(self, dimensions_to_prepend: Optional[tuple] = None) -> th.Tensor:
        """
        Sample a random uniform tensor from the space.
        If dimensions_to_prepend is not None, then the tensor will have these dimensions prepended.
        (e.g., for batch and time dimensions)

        Args:
            dimensions_to_prepend: The dimensions to prepend to the tensor.

        Returns:
            A random tensor from the space.
        """
        if self.low is None or self.high is None:
            raise ValueError("Cannot sample from an unbounded space.")
        if None in self.shape:
            raise ValueError("Cannot sample from a space with wildcard dimensions.")
        tensor_shape: tuple[int] = self.shape  # type: ignore # mypy thinks we can still have Nones
        if self.dtype.is_floating_point:
            random_tensor = th.rand(size=tensor_shape) * (self.high - self.low) + self.low
        else:
            high_plus_one = self.high + 1
            random_tensor = th.rand(size=tensor_shape) * (high_plus_one - self.low) + self.low
            random_tensor = th.floor(random_tensor)
        random_tensor = random_tensor.to(self.dtype)
        if dimensions_to_prepend is not None:
            assert isinstance(dimensions_to_prepend, tuple), f"Input dimensions_to_prepend {dimensions_to_prepend} is not a tuple."
            random_tensor = prepend_dimensions(random_tensor, dimensions_to_prepend)
        return random_tensor

    def get_preceding_dimensions(self, tensor: th.Tensor) -> tuple[int, ...]:
        """
        Return the preceding dimensions of a tensor that are not part of the space.
        Most commonly, these are the batch and time dimensions.

        E.g., if the space shape is (3, 4) and the tensor is (5, 3, 4), then the preceding dimensions are (5,).

        Args:
            tensor: The tensor to check.
        Returns:
            The preceding dimensions of the tensor as a tuple.
        """
        assert self.contains(tensor), f"Tensor {tensor} is not in the space {self}."
        return tuple(tensor.shape[: -self._ndim])


class TensorDictSpace:
    """
    Akin to TensorDict but for TensorSpaces.
    Holds a collection of TensorSpaces, each with a unique key. Operations are broadcast over keys.
    """

    def __init__(self, tensor_spaces: dict[str, TensorSpace]):
        assert isinstance(tensor_spaces, dict), f"Input tensor_spaces {tensor_spaces} is not a dict."
        for key, tensor_space in tensor_spaces.items():
            assert isinstance(key, str), f"Key {key} is not a string."
            assert isinstance(tensor_space, TensorSpace), f"Value {tensor_space} is not a TensorSpace."

        self._tensor_spaces = tensor_spaces

    @property
    def tensor_spaces(self) -> dict[str, TensorSpace]:
        return self._tensor_spaces

    def __getitem__(self, key: str) -> TensorSpace:
        return self.tensor_spaces[key]

    def __len__(self) -> int:
        return len(self.tensor_spaces)

    def __iter__(self) -> Iterator[str]:
        return self.keys()

    def keys(self) -> Iterator[str]:
        return iter(self.tensor_spaces.keys())

    def items(self) -> Iterator[Tuple[str, TensorSpace]]:
        return iter(self.tensor_spaces.items())

    def __repr__(self) -> str:
        return f"(TensorDictSpace tensor_spaces={self.tensor_spaces})"

    def _check_tensordict_keys(self, tensor_dict: TensorDict, allow_key_subset: bool = False) -> bool:
        """Check if input tensordict keys match the space keys, given the settings. Returns True if check passes, False otherwise."""
        tensor_dict_keys = set(tensor_dict.keys())
        tensor_spaces_keys = set(self.tensor_spaces.keys())
        if allow_key_subset:
            # TensorDict should not have extra keys, but can miss some of the space keys
            if not tensor_dict_keys.issubset(tensor_spaces_keys):
                return False
        else:
            # All keys should match
            if tensor_dict_keys != tensor_spaces_keys:
                return False
        return True

    def contains(self, tensor_dict: TensorDict, allow_key_subset: bool = False) -> bool:
        """
        Check if a TensorDict is in the space.
        TensorDict must have the matching keys (allow_key_subset = False) and each tensor must be in the corresponding TensorSpace.

        You can relax the requirement of all keys by setting allow_key_subset=False.
        If True, then only a subset of keys is required, but all tensors in the TensorDict must be in the corresponding TensorSpace.
        The TensorDict should also not have extra keys outside this space.

        Args:
            tensor_dict: The tensor dict to check.
            allow_key_subset: Whether to check that the keys match. If False, allow only a subset of keys.

        Returns:
            True if the tensor dict is in the space, False otherwise.
        """
        assert isinstance(tensor_dict, TensorDict), f"Input {tensor_dict} is not a tensordict."

        if not self._check_tensordict_keys(tensor_dict, allow_key_subset):
            return False

        for key, tensor in tensor_dict.items():
            if not self.tensor_spaces[key].contains(tensor):
                return False

        return True

    def is_subset_of(self, other: "TensorDictSpace") -> bool:
        """
        Check if this TensorDictSpace is a subset of another TensorDictSpace.
        See `TensorSpace.is_subset_of` for more details.
        This function repeats the check for each key in the TensorDictSpace, and returns True if all checks pass.

        Args:
            other: the assumed superspace.

        Returns:
            True if this space is a subset of the other space, False otherwise.
        """
        assert isinstance(other, TensorDictSpace), f"Input other {other} is not a TensorDictSpace."

        if set(self.tensor_spaces.keys()) != set(other.tensor_spaces.keys()):
            return False

        for key, tensor_space in self.tensor_spaces.items():
            if not tensor_space.is_subset_of(other.tensor_spaces[key]):
                return False

        return True

    def sample(self, dimensions_to_prepend: Optional[tuple] = None) -> TensorDict:
        """
        Sample a random tensor dict from the space.
        Raises if any of the spaces are unbounded or have wildcard dimensions.

        If dimensions_to_prepend is not None, then the tensor will have these dimensions prepended.
        (e.g., for batch and time dimensions).

        Args:
            dimensions_to_prepend: The dimensions to prepend to the tensor.

        Returns:
            TensorDict - a random tensor dict from the space.
        """
        assert dimensions_to_prepend is None or isinstance(dimensions_to_prepend, tuple), f"Input dimensions_to_prepend {dimensions_to_prepend} is not a tuple or None."
        if dimensions_to_prepend is None:
            dimensions_to_prepend = tuple()
        return TensorDict({key: tensor_space.sample(dimensions_to_prepend) for key, tensor_space in self.tensor_spaces.items()}, batch_size=dimensions_to_prepend)

    def get_preceding_dimensions(self, tensor_dict: TensorDict, allow_key_subset: bool = False) -> Optional[tuple[int, ...]]:
        """
        Return the preceding dimensions of the tensors. All tensors must have the same preceding dimensions.
        For TensorDicts, this corresponds to the "batch_dim" argument.

        If preceding dimensions are not the same, raises ValueError.

        You can relax the requirement of all keys by setting allow_key_subset=False.
        If True, then only a subset of keys is required, but all tensors in the TensorDict must be in the corresponding TensorSpace.
        The TensorDict should also not have extra keys outside this space.

        Args:
            tensor_dict: The tensor dict to check.
            allow_key_subset: Whether to check that the keys match. If False, allow only a subset of keys.

        Returns:
            The preceding dimensions of the tensors as a tuple.
        """
        assert isinstance(tensor_dict, TensorDict), f"Input tensor_dict {tensor_dict} is not a dict."

        if not self._check_tensordict_keys(tensor_dict, allow_key_subset):
            raise ValueError(f"TensorDict {tensor_dict} does not have the same keys as TensorDictSpace {self}.")

        preceding_dimension = None
        for key, tensor in tensor_dict.items():
            tensor_space = self.tensor_spaces[key]
            if not tensor_space.contains(tensor):
                raise ValueError(f"TensorDict {tensor_dict} does not have the same keys as TensorDictSpace {self}.")
            tensor_preceding_dimensions = tensor_space.get_preceding_dimensions(tensor)
            if preceding_dimension is None:
                preceding_dimension = tensor_preceding_dimensions

            if tensor_preceding_dimensions != preceding_dimension:
                raise ValueError(f"TensorDict {tensor_dict} has {tensor_preceding_dimensions} for preceding dimensions, expected {preceding_dimension}.")

        return preceding_dimension
