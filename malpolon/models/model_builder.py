from typing import Callable, Optional

import torch


class _ModelBuilder:
    providers = {}
    modifiers = {}

    def build_model(self, provider_name, model_name, model_args=[], model_kwargs={}, modifiers=None):
        provider = self.providers[provider_name]
        model = provider(model_name, *model_args, **model_kwargs)

        for modifier_name, modifier_kwargs in modifiers.items():
            modifier = self.modifiers[modifier_name]
            model = modifier(model, **modifier_kwargs)

        return model

    def register_provider(self, provider_name, provider):
        self.providers[provider_name] = provider

    def register_modifier(self, modifier_name, modifier):
        self.modifiers[modifier_name] = modifier


def torchvision_model_provider(model_name, *model_args, **model_kwargs):
    from torchvision import models

    model = getattr(models, model_name)
    model = model(*model_args, **model_kwargs)
    return model


def change_first_convolutional_layer_modifier(
    model: torch.nn.Module,
    num_input_channels: int,
    new_conv_layer_init_func: Optional[
        Callable[[torch.nn.Conv2d, torch.nn.Conv2d], None]
    ] = None,
) -> torch.nn.Module:
    """
    Removes the first convolutional layer of a model and replaces it by a new convolutional layer with the provided number of input channels.

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

    def find_conv_module(module):
        for child_name, child in module.named_children():
            if isinstance(child, torch.nn.Conv2d):
                return module, child_name
            else:
                res = find_conv_module(child)
                if res is not None:
                    return res

    submodule, layer_name = find_conv_module(model)
    old_layer = getattr(submodule, layer_name)

    new_layer = torch.nn.Conv2d(
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
    model: torch.nn.Module,
    num_outputs: int,
) -> torch.nn.Module:
    """
    Removes the last linear layer of a model and replaces it by a new dense layer with the provided number of outputs.

    Parameters
    ----------
    model: torch.nn.Module
        Model to adapt.
    num_outputs: integer
        Number of outputs of the new output layer.

    Returns
    -------
    model: torch.nn.Module
        Reference to model object given in input.
    """
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
        submodule = model
        layer_name = "fc"
    elif hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Linear):
        submodule = model
        layer_name = "classifier"
    elif (
        hasattr(model, "classifier")
        and isinstance(model.classifier, torch.nn.Sequential)
        and isinstance(model.classifier[-1], torch.nn.Linear)
    ):
        submodule = model.classifier
        layer_name = str(len(model.classifier) - 1)
    else:
        raise ValueError(
            "not supported architecture {}".format(model.__class__.__name__)
        )

    num_features = getattr(submodule, layer_name).in_features
    new_layer = torch.nn.Linear(num_features, num_outputs)
    setattr(submodule, layer_name, new_layer)

    return model


ModelBuilder = _ModelBuilder()

ModelBuilder.register_provider("torchvision", torchvision_model_provider)

ModelBuilder.register_modifier(
    "change_first_convolutional_layer",
    change_first_convolutional_layer_modifier,
)
ModelBuilder.register_modifier(
    "change_last_layer",
    change_last_layer_modifier,
)
