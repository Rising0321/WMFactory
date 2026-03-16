from typing import Any, Optional, Union

import torch as th
import pytorch_lightning as pl
from tensordict import TensorDict  # type: ignore # requires installing stubs for tensordict

# Using relative import so that this module is easier to move from this place to elsewhere
from .tensor_spaces import TensorSpace, TensorDictSpace


def assert_space_tuple_is_valid(space_tuple: tuple[TensorSpace]) -> None:
    """
    Checks that `space_tuple` is a valid tuple of TensorSpace objects.
    An empty tuple is valid.
    """
    if not isinstance(space_tuple, tuple):
        raise ValueError(f"Space tuple {space_tuple} is not a tuple")

    for space in space_tuple:
        assert isinstance(space, TensorSpace), f"Space {space} is not of type TensorSpace"


def assert_check_that_space_is_in_valid_spaces(space: TensorSpace, valid_spaces: tuple[TensorSpace]) -> None:
    """
    Checks that `space` is in `valid_spaces`, and raises an error if not.
    """
    if len(valid_spaces) == 0:
        raise ValueError(f"Valid spaces is empty, so {space} cannot be in it")

    for valid_space in valid_spaces:
        assert isinstance(valid_space, TensorSpace), f"Valid space {valid_space} is not of type TensorSpace"
        if space.is_subset_of(valid_space):
            return
    raise ValueError(f"Space {space} is not a subset of any of the valid spaces {valid_spaces}")


def assert_check_that_space_dict_is_in_valid_spaces(space_dict: TensorDictSpace, valid_spaces: tuple[TensorSpace]) -> None:
    """
    Checks that `space_dict` is a valid dictionary defining space name to TensorSpace, and that all spaces in the dict
    can part of `valid_spaces`.
    """
    if not isinstance(space_dict, TensorDictSpace):
        raise ValueError(f"Space dict {space_dict} is not a TensorDictSpace")

    for space_name, space in space_dict.items():
        space = space_dict[space_name]
        assert isinstance(space_name, str), f"Space name {space_name} is not a string"
        assert isinstance(space, TensorSpace), f"Space {space} is not of type TensorSpace"
        assert_check_that_space_is_in_valid_spaces(space, valid_spaces)


def assert_check_that_tensor_dict_is_valid_for_spaces(tensor_dict: TensorDict, spaces: TensorDictSpace, n_preceding_dims: Optional[int]) -> None:
    """
    Checks that input dictionary is a valid instantiation of the given spaces.
        - tensor_dict keys should match or be subset of spaces keys.
        - All values in tensor_dict should be in the corresponding space
        - All tensors have the same preceding dimensions (e.g., batch or batch and time)
        - Number of preceding dimensions is equal to `n_preceding_dims`, if provided

    Raises an error if any of these checks fail.

    Args:
        tensor_dict: TensorDict mapping space names to tensors.
        spaces: Dictionary mapping space names to TensorSpace
        n_preceding_dims: Number of preceding dimensions that all tensors should have (if None, this is not checked)
    """
    if not isinstance(tensor_dict, TensorDict):
        raise ValueError("Input dictionaries should be instances of tensordict.TensorDict")

    if not set(tensor_dict.keys()).issubset(set(spaces.keys())):
        raise ValueError(f"Input dict keys {tensor_dict.keys()} does not match or is not a subset of space keys {spaces.keys()}")

    expected_preceding_dims = None
    for tensor_name, tensor_tensor in tensor_dict.items():
        assert isinstance(tensor_name, str), f"Input name {tensor_name} is not a string"
        assert isinstance(tensor_tensor, th.Tensor), f"Tensor {tensor_tensor} is not a tensor"

        space = spaces[tensor_name]
        if not space.contains(tensor_tensor):
            raise ValueError(f"Tensor {tensor_name} is not in space {space}")

        preceding_dims = space.get_preceding_dimensions(tensor_tensor)

        if n_preceding_dims is not None and len(preceding_dims) != n_preceding_dims:
            raise ValueError(f"Tensor {tensor_name} has {len(preceding_dims)} preceding dimensions, but expected {n_preceding_dims}")

        if expected_preceding_dims is None:
            expected_preceding_dims = preceding_dims
        else:
            assert preceding_dims == expected_preceding_dims, f"Tensor {tensor_name} has preceding dims {preceding_dims}, but expected {expected_preceding_dims}"


class PredictorBase(pl.LightningModule):
    """
    Base class for "predictor" torch modules, which are used to predict future states from a context.

    Args:
        context_space: A TensorDictSpace defining the space for each context modality.
                       Must have at least one context modality.
        condition_space: A dictionary mapping condition names to their spaces.
                         This may be empty or None if there are no conditions.
    """

    # This is a dictionary of keyword arguments that can be used to create this class
    # during testing/quick debugging (e.g., minimal model size)
    __DEBUG_CREATION_KWARGS__: dict[str, Any] = dict()

    # These class attributes are used to define the acceptable spaces for the predictor.
    # They are used in the `__init__` method to check that the spaces user is trying to use are valid for this predictor
    _acceptable_context_spaces: Optional[Union[TensorSpace, tuple[TensorSpace]]] = None
    _acceptable_condition_spaces: Optional[Union[tuple[TensorSpace], tuple[TensorSpace]]] = None

    def __init__(self, context_space: TensorDictSpace, condition_space: Optional[TensorDictSpace] = None):
        if condition_space is None:
            condition_space = TensorDictSpace(dict())
        assert_check_that_space_dict_is_in_valid_spaces(context_space, self.acceptable_context_spaces)
        assert_check_that_space_dict_is_in_valid_spaces(condition_space, self.acceptable_condition_spaces)
        assert len(context_space) > 0, "There must be at least one context encoder space"
        super().__init__()

        self.context_space = context_space
        self.condition_space = condition_space

    @property
    def acceptable_context_spaces(self) -> tuple[TensorSpace]:
        self_class = self.__class__
        _acceptable_context_spaces = self_class._acceptable_context_spaces
        assert (
            _acceptable_context_spaces is not None
        ), f"Class {self_class} has no _acceptable_context_spaces class property defined. This should be a tuple of TensorSpace"
        if not isinstance(_acceptable_context_spaces, tuple):
            _acceptable_context_spaces = (_acceptable_context_spaces,)
        try:
            assert_space_tuple_is_valid(_acceptable_context_spaces)
        except Exception as e:
            raise AssertionError(f"Class {self_class} has an invalid _acceptable_context_spaces class property. See above for details") from e
        return _acceptable_context_spaces

    @property
    def acceptable_condition_spaces(self) -> tuple[TensorSpace]:
        self_class = self.__class__
        _acceptable_condition_spaces = self_class._acceptable_condition_spaces
        assert (
            _acceptable_condition_spaces is not None
        ), f"Class {self_class} has no _acceptable_condition_spaces class property defined. This should be a tuple of TensorSpace"
        if not isinstance(_acceptable_condition_spaces, tuple):
            _acceptable_condition_spaces = (_acceptable_condition_spaces,)
        try:
            assert_space_tuple_is_valid(_acceptable_condition_spaces)
        except Exception as e:
            raise AssertionError(f"Class {self_class} has an invalid _acceptable_condition_spaces class property. See above for details") from e
        return _acceptable_condition_spaces

    def assert_check_context_tensordict_is_valid(self, context_dict: TensorDict) -> None:
        """
        Checks that context tensordict is a valid set of modalities for this predictor:
            - context_dict should have all the modalities that this predictor expects
            - All the tensors should be contained in the spaces that this predictor expects
            - Preceding dimensions of all tensors should be the same, and have two die dimensions (batch and time)

        Raises an error if the context dictionary is invalid.
        """
        try:
            assert_check_that_tensor_dict_is_valid_for_spaces(context_dict, self.context_space, n_preceding_dims=2)
        except Exception as e:
            raise ValueError(f"Context TensorDict {context_dict} is not valid for this predictor. See above exception for more info.") from e

    def assert_check_condition_dict_is_valid(self, condition_dict: TensorDict) -> None:
        """
        Checks that input dictionary is a valid set of modalities for this predictor:
            - context_dict should have all the modalities that this predictor expects
            - All the tensors should be contained in the spaces that this predictor expects
            - Conditions should _only_ have batch dimension

        Raises an error if the input is invalid.
        """
        try:
            assert_check_that_tensor_dict_is_valid_for_spaces(condition_dict, self.condition_space, n_preceding_dims=1)
        except Exception as e:
            raise ValueError(f"Condition TensorDict {condition_dict} is not valid for this predictor. See above exception for more info.") from e
