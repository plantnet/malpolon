from torch import nn


def change_last_layer(model, n_classes):
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
        return model.fc
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        num_ftrs = model.classifier
        model.classifier = nn.Linear(num_ftrs, n_classes)
        return model.classifier
    elif (
        hasattr(model, "classifier")
        and isinstance(model.classifier, nn.Sequential)
        and isinstance(model.classifier[-1], nn.Linear)
    ):
        num_ftrs = model.classifier[-1]
        model.classifier[-1] = nn.Linear(num_ftrs, n_classes)
        return model.classifier[-1]
    else:
        raise ValueError(
            "not supported architecture {}".format(model.__class__.__name__)
        )
