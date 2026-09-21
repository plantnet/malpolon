"""Model-specific image preprocessing used consistently across dataset splits."""

from __future__ import annotations

from torchvision import transforms

_SPECS = {
    "resnet18": (256, 224), "resnet50": (232, 224), "convnext": (236, 224), "vgg16": (256, 224),
    "vitb32": (224, 224), "mobilenet_v3": (232, 224), "inception_v3": (342, 299),
    "dinov2_vits14": (520, 518), "dinov2_PN22M": (540, 512),
}
# These statistics intentionally match the original baseline, rather than the
# torchvision defaults, to keep comparable input scaling across experiments.
_IMAGENET_MEAN, _IMAGENET_STD = (0.442, 0.469, 0.326), (0.229, 0.224, 0.225)
_PN22M_MEAN, _PN22M_STD = (0.485, 0.456, 0.406), (0.235, 0.221, 0.232)


def make_transforms(model_name: str):
    """Return stochastic training and deterministic evaluation transforms.

    The resize/crop pair is backbone-specific. The PN22M crop retains the
    original larger field of view used to avoid image watermarks.
    """
    try:
        resize, crop = _SPECS[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported model '{model_name}'") from error
    mean, std = (_PN22M_MEAN, _PN22M_STD) if model_name == "dinov2_PN22M" else (_IMAGENET_MEAN, _IMAGENET_STD)
    normalise = transforms.Normalize(mean=mean, std=std)
    base = [transforms.Resize(resize), transforms.CenterCrop(crop), transforms.ToTensor(), normalise]
    training = transforms.Compose(
        base[:2]
        + [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
        + base[2:]
    )
    evaluation = transforms.Compose(base)
    return training, evaluation
