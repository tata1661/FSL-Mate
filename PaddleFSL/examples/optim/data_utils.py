"""Data Utils for Meta Optimzations Algorithms"""
from __future__ import annotations
from typing import Tuple, Dict
import paddlefsl
from paddlefsl.datasets.cv_dataset import CVDataset


def load_datasets(name: str) -> Tuple[CVDataset, CVDataset, CVDataset]:
    """load CV Dataset by name, which can be omniglot, miniimagenet, or cifar10

    Args:
        name (str): the name of datasets

    Returns:
        Tuple[CVDataset, CVDataset, CVDataset]: train, dev, test dataset
    """
    datasets_map: Dict[str, CVDataset] = {
        "omniglot": (
            paddlefsl.datasets.Omniglot(mode='train', image_size=(28, 28)),
            paddlefsl.datasets.Omniglot(mode='valid', image_size=(28, 28)),
            paddlefsl.datasets.Omniglot(mode='test', image_size=(28, 28))
        ),
        # "miniimagenet": (
        #     paddlefsl.datasets.MiniImageNet(mode='train'),
        #     paddlefsl.datasets.MiniImageNet(mode='valid'),
        #     paddlefsl.datasets.MiniImageNet(mode='test')
        # ),
        # "cifarfs": (
        #     paddlefsl.datasets.CifarFS(mode='train', image_size=(28, 28)),
        #     paddlefsl.datasets.CifarFS(mode='valid', image_size=(28, 28)),
        #     paddlefsl.datasets.CifarFS(mode='test', image_size=(28, 28))
        # ),
        # "fc100": (
        #     paddlefsl.datasets.FC100(mode='train'),
        #     paddlefsl.datasets.FC100(mode='valid'),
        #     paddlefsl.datasets.FC100(mode='test')
        # ),
        # "cub": (
        #     paddlefsl.datasets.CubFS(mode='train'),
        #     paddlefsl.datasets.CubFS(mode='valid'),
        #     paddlefsl.datasets.CubFS(mode='test')
        # )
    }
    if name not in datasets_map:
        names = ",".join(list(datasets_map.keys()))
        raise ValueError(f"{name} is not a valid dataset name, which should be in {names}")

    return datasets_map[name]
