import torch
from torchvision import models

from malpolon.models.standard_classification_models import change_first_layer


def test_change_first_layer():
    model_names = [
        ("resnet18", 224),
        ("inception_v3", 299),
        ("densenet121", 224),
        ("efficientnet_b0", 224),
        ("vit_b_16", 224),
    ]

    num_input_channels = 10

    for model_name, image_size in model_names:
        model_func = getattr(models, model_name)
        model = model_func()

        change_first_layer(model, num_input_channels)

        x = torch.rand(2, num_input_channels, image_size, image_size, dtype=torch.float32)
        model(x)
