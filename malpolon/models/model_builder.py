from __future__ import annotations
from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from typing import Any, Callable, Optional

    Provider = Callable[..., nn.Module]
    Modifier = Callable[..., nn.Module]


class _ModelBuilder:
    providers: dict[str, Provider] = {}
    modifiers: dict[str, Modifier] = {}

    def build_model(
        self,
        provider_name: str,
        model_name: str,
        model_args: list = [],
        model_kwargs: dict = {},
        modifiers: dict[str, dict[str, Any]] = {},
    ) -> nn.Module:
        provider = self.providers[provider_name]
        model = provider(model_name, *model_args, **model_kwargs)

        for modifier_name, modifier_kwargs in modifiers.items():
            modifier = self.modifiers[modifier_name]
            model = modifier(model, **modifier_kwargs)

        return model

    def register_provider(self, provider_name: str, provider: Provider) -> None:
        self.providers[provider_name] = provider

    def register_modifier(self, modifier_name: str, modifier: Modifier) -> None:
        self.modifiers[modifier_name] = modifier


def torchvision_model_provider(model_name: str, *model_args: Any, **model_kwargs: Any) -> nn.Module:
    from torchvision import models

    model = getattr(models, model_name)
    model = model(*model_args, **model_kwargs)
    return model


def _find_module_of_type(module: nn.Module, module_type: type, order: str) -> tuple[nn.Module, str]:
    if order == "first":
        modules = module.named_children()
    elif order == "last":
        modules = reversed(list(module.named_children()))
    else:
        raise ValueError(
            "order must be either 'first' or 'last', given {}".format(order)
        )

    for child_name, child in modules:
        if isinstance(child, module_type):
            return module, child_name
        else:
            res = _find_module_of_type(child, module_type, order)
            if res is not None:
                return res


def change_first_convolutional_layer_modifier(
    model: nn.Module,
    num_input_channels: int,
    new_conv_layer_init_func: Optional[
        Callable[[nn.Conv2d, nn.Conv2d], None]
    ] = None,
) -> nn.Module:
    """
    Removes the first registered convolutional layer of a model and replaces it by a new convolutional layer with the provided number of input channels.

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
) -> nn.Module:
    """
    Removes the last registered linear layer of a model and replaces it by a new dense layer with the provided number of outputs.

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
    submodule, layer_name = _find_module_of_type(model, nn.Linear, "last")
    old_layer = getattr(submodule, layer_name)

    num_features = old_layer.in_features
    new_layer = nn.Linear(num_features, num_outputs)
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
