from typing import Any, Type, Callable, Optional

import pytorch_lightning as pl


class LightningModuleCreationArgs:
    """
    A creator class for holding arguments to define creating/loading a PL Module, either from scratch or from a checkpoint.

    Three combinations are possible:
        - `pl_class` is provided, create a new module. Pass in `**pl_kwargs` (defaults to empty dict) as kwargs.
        - `pl_class` and `pl_checkpoint_path` are provided, load a module from a checkpoint.
        - `pl_checkpoint_path`, `pl_class` and `pl_kwargs` are provided, load a module from a checkpoint, and overwrite checkpoint arguments with kwargs.

    Additionally, there is `pl_stored_params_override`, which is a dictionary we use to update the module's hyperparameters when saving.

    Motivation:
        Pytorch Lightning checkpoint saving stores the arguments passed to __init__ of the modules, and uses them to recreate the object.
        If we are training a model that uses other PL modules, we usually want to pass in a path to a module checkpoint to restore module from.
            (e.g., a bigger module uses pretrained encoder, and loads it from a checkpoint path).
        However, when we save the module and try to restore it, it tries to use the same checkpoint path to load up the encoder, which probably doesn't exist.
            (e.g., if we trained the bigger module on a different machine, and the current machine does not have the encoder checkpoint).
        Ideally, we would want the bigger module contain everything to recreate the encoder, and not rely on a checkpoint path.

        This CreationArgs class aims to solve this by initially loading the encoder from a checkpoint path while training, but then
        storing the arguments used to create the encoder in the bigger module, so that we can recreate the encoder from scratch when loading the bigger module.

        `pl_stored_params_override` can be used to replace custom hyperparameters of the module when saving, e.g. if we want to remove the checkpoint path.
    """

    def __init__(
        self, pl_class: Type[pl.LightningModule], pl_checkpoint_path: Optional[str] = None, pl_stored_params_override: Optional[dict[str, Any]] = None, **pl_kwargs
    ):
        assert pl_class is not None, "Must provide a class for the PL module."
        self.pl_class = pl_class
        self.pl_checkpoint_path = pl_checkpoint_path
        self.pl_kwargs = pl_kwargs
        self.pl_stored_params_override = pl_stored_params_override

    @classmethod
    def from_dict(self, config: dict[str, Any], class_name_to_class_fn: Callable[[str], Type[pl.LightningModule]]) -> "LightningModuleCreationArgs":
        """
        Create a LightningModuleCreationArgs object from a config dict.
        The config dictionary should have the following entries:
            - `__class_name__`: str, name of the PL module class to create. This is passed to the `class_name_to_class_fn` function.
            - `__checkpoint_path__`: The checkpoint path to load the PL module from (optional)
            - Rest of the arguments are passed as **kwargs to the constructor

        Args:
            config: The config dictionary.
            class_name_to_class_fn: A function, mapping class names to classes.
        Returns:
            A LightningModuleCreationArgs object.
        """
        assert "__class_name__" in config, "Must provide a class name for the PL module as `__class_name__`."
        pl_class = class_name_to_class_fn(config["__class_name__"])
        checkpoint_path = config.get("__checkpoint_path__", None)
        stored_params_override = config.get("__stored_hparams_override__", None)

        kwargs = config.copy()
        del kwargs["__class_name__"]
        if "__checkpoint_path__" in kwargs:
            del kwargs["__checkpoint_path__"]
        if "__stored_hparams_override__" in kwargs:
            del kwargs["__stored_hparams_override__"]

        return LightningModuleCreationArgs(pl_class=pl_class, pl_checkpoint_path=checkpoint_path, pl_stored_params_override=stored_params_override, **kwargs)

    def create_module(self, remove_checkpoint_path: bool = True, **kwargs) -> pl.LightningModule:
        """
        Create the PL module based on arguments:
            - If `pl_checkpoint_path` is provided, load a module from a checkpoint.
            - Otherwise, create a new module with `pl_class` and `pl_kwargs`.

        If `remove_checkpoint_path` is True, then the creation kwargs will be updated to match the checkpoint arguments,
        and checkpoint path will be removed. This is to make loading nested models easier.

        **kwargs will be used to overwrite `pl_kwargs` if both are provided.
        """
        pl_kwargs = self.pl_kwargs.copy()
        pl_kwargs.update(kwargs)
        if self.pl_checkpoint_path is not None:
            pl_module = self.pl_class.load_from_checkpoint(self.pl_checkpoint_path, **pl_kwargs)
            if remove_checkpoint_path:
                self.update_creation_args_to_match_module(pl_module)
        else:
            # "remove_checkpoint_path" does not have an effect here, as all settings should be in the kwargs already.
            pl_module = self.pl_class(**pl_kwargs)
            # However, if we args we want to override the stored hparams, we can do that here.
            if self.pl_stored_params_override is not None:
                if pl_module.hparams is None:
                    # We have to make sure hyperparameters are stored in the module, otherwise they won't be saved.
                    pl_module.save_hyperparameters()
                self.pl_kwargs.update(self.pl_stored_params_override)
                pl_module.hparams.update(self.pl_kwargs)
                self.pl_stored_params_override = None

        return pl_module

    def update_creation_args_to_match_module(self, module: pl.LightningModule) -> None:
        """
        Update this object's creation arguments to match those of the saved model.
        If this module already has kwargs, then these will be used to overwrite the checkpoint arguments.
        Also removes the checkpoint path, if it exists.

        This is intended to be used to update the creation arguments of a model loaded from a checkpoint,
        so that when creating nested models, we keep all hyperparameters in a single place
        """
        assert isinstance(module, self.pl_class), f"Module is not of type {self.pl_class}"
        module_hparams = module.hparams
        if self.pl_kwargs is None:
            self.pl_kwargs = dict()
        module_hparams.update(self.pl_kwargs)
        # More of a typing thing, but lets make sure all entries are valid arguments
        # for model (i.e. keys are strings)
        new_pl_kwargs = dict()
        for key, value in module_hparams.items():
            assert isinstance(key, str), f"Key {key} in module hyperameters is not a string"
            new_pl_kwargs[key] = value
        self.pl_kwargs = new_pl_kwargs
        self.pl_checkpoint_path = None
