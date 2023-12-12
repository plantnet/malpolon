import timm
import torch
from torchvision import models

from malpolon.models.model_builder import (
    change_first_convolutional_layer_modifier, change_last_layer_modifier,
    timm_model_provider, torchvision_model_provider)


def test_change_first_convolutional_layer():
    model_names = [
        ("vgg11", 224),
        ("resnet18", 224),
        ("inception_v3", 299),
        ("densenet121", 224),
        ("efficientnet_b0", 224),
        ("vit_b_16", 224),
    ]

    num_input_channels = 10

    for model_name, image_size in model_names:
        model_func = getattr(models, model_name)
        if model_name == "inception_v3":
            model = model_func(init_weights=False)
        else:
            model = model_func()

        change_first_convolutional_layer_modifier(model, num_input_channels)

        x = torch.rand(
            2, num_input_channels, image_size, image_size, dtype=torch.float32
        )
        model(x)

def test_change_last_layer():
    model_names = [
        ("vgg11", 224),
        ("resnet18", 224),
        ("inception_v3", 299),
        ("densenet121", 224),
        ("efficientnet_b0", 224),
        ("vit_b_16", 224),
    ]

    num_outputs = 10

    for model_name, image_size in model_names:
        model_func = getattr(models, model_name)
        if model_name == "inception_v3":
            model = model_func(init_weights=False)
        else:
            model = model_func()

        change_last_layer_modifier(model, num_outputs)

        x = torch.rand(2, 3, image_size, image_size, dtype=torch.float32)
        y = model(x)

        if isinstance(y, models.inception.InceptionOutputs):
            y = y.logits

        assert y.shape[-1] == num_outputs


def test_torchvision_model_provider():
    model_kwargs = {'weights': None}
    model_name = 'resnet18'
    model = torchvision_model_provider(model_name, **model_kwargs)
    assert isinstance(model, models.resnet.ResNet)


def test_timm_model_provider():
    model_kwargs = {'pretrained': True}
    model_name = 'resnet18'
    model = timm_model_provider(model_name, **model_kwargs)
    assert isinstance(model, timm.models.resnet.ResNet)
