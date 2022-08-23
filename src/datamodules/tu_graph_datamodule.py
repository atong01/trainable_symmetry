from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9, TUDataset
from torch_geometric.transforms import Compose, Distance

from .components.fast_scatter import FastScatterTransform
from .components.transforms import ClusteringCoefficient, Eccentricity, Standardize


def get_transform(name, device, **kwargs):
    if name == "eccentricity":
        transform = Eccentricity()
    elif name == "clustering_coefficient":
        transform = ClusteringCoefficient()
    elif name == "scatter":
        transform = Compose([Eccentricity(), ClusteringCoefficient(cat=True)])
    elif name == "scatter_cat":
        transform = Compose([Eccentricity(cat=True), ClusteringCoefficient(cat=True)])
    elif name == "scatter_cat_norm":
        transform = Compose(
            [
                Eccentricity(cat=True),
                ClusteringCoefficient(cat=True),
                Standardize(cat=False),
            ]
        )
    elif name == "fast_scatter":
        transform = Compose(
            [
                Eccentricity(),
                ClusteringCoefficient(cat=True),
                FastScatterTransform(device, **kwargs),
            ]
        )
    elif name == "fast_scatter_noagg":
        transform = Compose(
            [
                Eccentricity(),
                ClusteringCoefficient(cat=True),
                FastScatterTransform(device, agg=None),
            ]
        )
    elif name == "fast_scatter_cat":
        transform = Compose(
            [
                Eccentricity(cat=True),
                ClusteringCoefficient(cat=True),
                FastScatterTransform(device),
            ]
        )
    elif name == "none" or name is None:
        transform = None
    else:
        raise NotImplementedError("Unknown transform %s" % name)
    print(transform)
    return transform


def frac_to_num(arr, total):
    """convert train_val_test split expressed as fractions to the total"""
    # Good approximation
    x = np.floor(np.array(arr) * total)
    tot = x.sum()
    x[2] += total - tot
    x = x.astype(int)
    return tuple(x)


class TUGraphDataModule(LightningDataModule):

    TU_DATASETS = [
        "NCI1",
        "NCI109",
        "DD",
        "PROTEINS",
        "MUTAG",
        "PTC_MR",
        "ENZYMES",
        "REDDIT-BINARY",
        "REDDIT-MULTI-12K",
        "IMDB-BINARY",
        "IMDB-MULTI",
        "COLLAB",
        "REDDIT-MULTI-5K",
    ]

    def __init__(
        self,
        dataset: str,
        transform: Optional[str] = None,
        transform_args: Optional[dict] = None,
        data_dir: str = "data/",
        train_val_test_split: Union[
            Tuple[int, int, int], Tuple[float, float, float]
        ] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        if transform is None:
            transform = "none"
        if transform_args is None:
            transform_args = {}

        self.transform = get_transform(transform, "cpu", **transform_args)

        # print("Getting dataset %s" % args["dataset"])

        dataset = TUDataset(
            root=f"{data_dir}/{transform}_{transform_args}",
            name=dataset,
            pre_transform=self.transform,
            use_node_attr=True,
        )
        self.train_val_test_split = frac_to_num(train_val_test_split, len(dataset))
        self.num_classes = torch.max(dataset.data["y"]) + 1
        self.input_dim = dataset.data.x.shape[1]
        self.data_train, self.data_val, self.data_test = random_split(
            dataset,
            self.train_val_test_split,
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "tu_graph.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
