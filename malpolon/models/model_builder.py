"""This module provides classes to build your PyTorch models.

Classes listed in this module allow to select a model from your
provider (timm, torchvision...), retrieve it with or without
pre-trained weights, and modify it by adding or removing layers.

Author: Titouan Lorieul <titouan.lorieul@gmail.com>
        Theo Larcher <theo.larcher@inria.fr>

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import timm
from torch import nn
from torchvision import models

from malpolon.models.custom_models.glc2024_multimodal_ensemble_model import \
    MultimodalEnsemble

if TYPE_CHECKING:
    from typing import Any, Callable, Optional

    Provider = Callable[..., nn.Module]
    Modifier = Callable[..., nn.Module]

MALPOLON_MODELS = {'glc24_multimodal_ensemble': MultimodalEnsemble, }


class _ModelBuilder:
    """General class to build models."""
    providers: dict[str, Provider] = {}
    modifiers: dict[str, Modifier] = {}

    def build_model(
        self,
        provider_name: str,
        model_name: str,
        model_args: list = [],
        model_kwargs: dict = {},
        modifiers: dict[str, Optional[dict[str, Any]]] = {},
    ) -> nn.Module:
        """Return a built model with the given provider and modifiers.

        Parameters
        ----------
        provider_name : str
            source of the model's provider, valid values are:
            [`timm`, `torchvision`]
        model_name : str
            name of the model to retrieve from the provider
        model_args : list, optional
            model arguments to pass on when building it, by default []
        model_kwargs : dict, optional
            model kwargs, by default {}
        modifiers : dict[str, Optional[dict[str, Any]]], optional
            modifiers to call on the model after it is built,
            by default {}

        Returns
        -------
        nn.Module
            built and mofified model
        """
        provider = self.providers[provider_name]
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        model = provider(model_name, *model_args, **model_kwargs)

        for modifier_name, modifier_kwargs in modifiers.items():
            modifier_kwargs = modifier_kwargs or {}
            modifier = self.modifiers[modifier_name]
            model = modifier(model, **modifier_kwargs)

        return model

    def register_provider(self, provider_name: str, provider: Provider) -> None:
        """Register a provider to the model builder.

        Parameters
        ----------
        provider_name : str
            name of the provider, valid values are:
            [`timm`, `torchvision`]
        provider : Provider
            callable provider function
        """
        self.providers[provider_name] = provider

    def register_modifier(self, modifier_name: str, modifier: Modifier) -> None:
        """Register a modifier to the model builder.

        Parameters
        ----------
        modifier_name : str
            name of the modifier, valid values are:
            [`change_first_convolutional_layer`, `change_last_layer`, `change_last_layer_to_identity`]
        modifier : Modifier
            modifier callable function
        """
        self.modifiers[modifier_name] = modifier


def torchvision_model_provider(
    model_name: str, *model_args: Any, **model_kwargs: Any
) -> nn.Module:
    """Return a model from torchvision's library.

    This method uses tochvision's API to retrieve a model from its
    library.

    Parameters
    ----------
    model_name : str
        name of the model to retrieve from torchvision's library

    Returns
    -------
    nn.Module
        model object
    """
    model = getattr(models, model_name)
    model = model(*model_args, **model_kwargs)
    return model


def timm_model_provider(
    model_name: str, *model_args: Any, **model_kwargs: Any
) -> nn.Module:
    """Return a model from timm's library.

    This method uses timm's API to retrieve a model from its library.

    Parameters
    ----------
    model_name : str
        name of the model to retrieve from timm's library

    Returns
    -------
    nn.Module
        model object

    Raises
    ------
    ValueError
        if the model name is not listed in TIMM's library
    """
    available_models = timm.list_models()
    if model_name in available_models:
        model = timm.create_model(model_name, *model_args, **model_kwargs)
    else:
        raise ValueError(
            f"Model name is not listed in TIMM's library. Please choose a model"
            f" amongst the following list: {available_models}"
        )
    return model


def malpolon_model_provider(
    model_name: str, *model_args: Any, **model_kwargs: Any
) -> nn.Module:
    """Return a model from Malpolon's models list.

    This method uses Malpolon's internal model listing to retrieve a
    model.

    Parameters
    ----------
    model_name : str
        name of the model to retrieve from torchvision's library

    Returns
    -------
    nn.Module
        model object
    """
    model = MALPOLON_MODELS[model_name]
    model = model(*model_args, **model_kwargs)
    return model


def _find_module_of_type(
    module: nn.Module, module_type: type, order: str
) -> tuple[nn.Module, str]:
    """Find the first or last module of a given type in a module.

    Parameters
    ----------
    module : nn.Module
        torch module to search in (_e.g.: torch model_)
    module_type : type
        module type to search for (_e.g.: nn.Conv2d_)
    order : str
        order to search for the module, valid values are:
        [`first`, `last`]

    Returns
    -------
    tuple[nn.Module, str]
        module and its name

    Raises
    ------
    ValueError
        if the order is not valid
    """
    if order == "first":
        modules = module.named_children()
    elif order == "last":
        modules = reversed(list(module.named_children()))
    else:
        raise ValueError(
            f"order must be either 'first' or 'last', given {order}"
        )

    for child_name, child in modules:
        if isinstance(child, module_type):
            return module, child_name
        res = _find_module_of_type(child, module_type, order)
        if res[1] != "":
            return res

    return module, ""


def change_first_convolutional_layer_modifier(
    model: nn.Module,
    num_input_channels: int,
    new_conv_layer_init_func: Optional[Callable[[nn.Conv2d, nn.Conv2d], None]] = None,
) -> nn.Module:
    """Remove the first registered convolutional layer of a model and replaces it by a new convolutional layer with the provided number of input channels.

    Parameters
    ----------
    model: torch.nn.Module
        Model to adapt.
    num_input_channels: integer
        Number of input channels, used to update the first convolutional layer.
    new_conv_layer_init_func: callable
        Function defining how to initialize the new convolutional layer.

    Returns
    -------
    model: torch.nn.Module
        Newly created last dense classification layer.
    """
    submodule, layer_name = _find_module_of_type(model, nn.Conv2d, "first")
    old_layer = getattr(submodule, layer_name)

    new_layer = nn.Conv2d(
        num_input_channels,
        out_channels=old_layer.out_channels,
        kernel_size=old_layer.kernel_size,
        stride=old_layer.stride,
        padding=old_layer.padding,
        dilation=old_layer.dilation,
        groups=old_layer.groups,
        bias=old_layer.bias is not None,
        padding_mode=old_layer.padding_mode,
        device=old_layer.weight.device,
        dtype=old_layer.weight.dtype,
    )

    if new_conv_layer_init_func:
        new_conv_layer_init_func(old_layer, new_layer)

    setattr(submodule, layer_name, new_layer)

    return model


def change_last_layer_modifier(
    model: nn.Module,
    num_outputs: int,
    flatten: bool = False,
) -> nn.Module:
    """Remove the last registered linear layer of a model and replaces it by a new dense layer with the provided number of outputs.

    Parameters
    ----------
    model: torch.nn.Module
        Model to adapt.
    num_outputs: integer
        Number of outputs of the new output layer.
    flatten: boolean
        If True, adds a nn.Flatten layer to squeeze the last dimension. Can be useful when num_outputs=1.

    Returns
    -------
    model: torch.nn.Module
        Reference to model object given in input.
    """
    submodule, layer_name = _find_module_of_type(model, nn.Linear, "last")
    old_layer = getattr(submodule, layer_name)

    num_features = old_layer.in_features
    new_layer = nn.Linear(num_features, num_outputs)

    if flatten:
        new_layer = nn.Sequential(
            new_layer,
            nn.Flatten(0, -1),
        )

    setattr(submodule, layer_name, new_layer)

    return model


def change_last_layer_to_identity_modifier(model: nn.Module) -> nn.Module:
    """Remove the last  linear layer of a model and replaces it by an nn.Identity layer.

    Parameters
    ----------
    model: torch.nn.Module
        Model to adapt.

    Returns
    -------
    num_features: int
        Size of the feature space.
    """
    submodule, layer_name = _find_module_of_type(model, nn.Linear, "last")

    new_layer = nn.Identity()
    setattr(submodule, layer_name, new_layer)

    return model


ModelBuilder = _ModelBuilder()

ModelBuilder.register_provider("torchvision", torchvision_model_provider)
ModelBuilder.register_provider("timm", timm_model_provider)
ModelBuilder.register_provider("malpolon", malpolon_model_provider)

ModelBuilder.register_modifier(
    "change_first_convolutional_layer",
    change_first_convolutional_layer_modifier,
)
ModelBuilder.register_modifier(
    "change_last_layer",
    change_last_layer_modifier,
)
ModelBuilder.register_modifier(
    "change_last_layer_to_identity",
    change_last_layer_to_identity_modifier,
)
