from pytorch_lightning.callbacks import Callback


def str_object(obj: object) -> str:
    """
    Formats an object to printing by returning a string containing the
    class name and attributes (both name and values)

    Parameters
    ----------
    obj: object to print.

    Returns
    -------
    str: string containing class name and attributes.
    """

    class_name = obj.__class__.__name__
    attributes = obj.__dict__

    filtered_attributes = []
    for key, val in attributes.items():
        # Test if its a private attribute
        if not key.startswith("_"):
            # Test if is not builtin type
            if hasattr(val, "__module__"):
                val = "<object>"
            filtered_attributes.append((key, val))

    formatted_attributes = ", ".join(
        map(lambda x: "{}={}".format(*x), filtered_attributes)
    )
    return "{}(\n    {}\n)".format(class_name, formatted_attributes)


class Summary(Callback):
    """
    FIXME handle multi validation data loaders, combined datasets
    """
    def _log_data_loading_summary(self, data_loader, split):
        if split == "Train":
            dataset = data_loader.dataset.datasets
        else:
            dataset = data_loader.dataset

        from torch.utils.data import Subset
        if isinstance(dataset, Subset):
            dataset = dataset.dataset

        print("{} dataset: {}".format(split, dataset))
        print("{} set size: {}".format(split, len(dataset)))

        if split == "Train" and hasattr(dataset, "n_classes"):
            print("Number of classes: {}".format(dataset.n_classes))

        if hasattr(dataset, "transform"):
            print("{} data transformations: {}".format(split, dataset.transform))

        if hasattr(dataset, "target_transform"):
            print("{} data target transformations: {}".format(split, dataset.target_transform))

        print("{} data sampler: {}".format(split, str_object(data_loader.sampler)))

        if hasattr(data_loader, "loaders"):
            batch_sampler = data_loader.loaders.batch_sampler
        else:
            batch_sampler = data_loader.batch_sampler
        print("{} data batch sampler: {}".format(split, str_object(batch_sampler)))

    def on_train_start(self, trainer, model):
        print("\n# Model specification")
        print(model.model)
        print(model.loss)
        print(model.optimizer)
        print("Metrics: {}".format(model.metrics))

        print("\n# Data loading information")
        print("\n## Training data")
        self._log_data_loading_summary(trainer.train_dataloader, "Train")

        print("\n## Validation data")
        self._log_data_loading_summary(trainer.val_dataloaders[0], "Validation")
