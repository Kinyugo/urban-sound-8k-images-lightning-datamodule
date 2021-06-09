import os
from argparse import ArgumentParser
from typing import Callable, Optional, Tuple, Union

import gdown
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder


class UrbanSound8kImagesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        train_transforms: Callable = None,
        val_transforms: Callable = None,
        test_transforms: Callable = None,
        image_size: Union[Tuple[int, int], int] = (256, 256),
        num_workers: int = 1,
        seed: int = 0,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        normalize: bool = True,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.image_size = image_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.normalize = normalize

        # setup transforms with default values
        self._init_transforms()

    def prepare_data(self) -> None:
        # download data
        url = "https://drive.google.com/uc?id=1y6E94oielvL1fiLbXydMSIvA8JYAoOcL"
        output = os.path.join(self.data_dir, "urbansound8k_images.tar.gz")
        md5_hash = "cd0775b732c94705377a2871cd0a3d8e"

        gdown.cached_download(url, output, md5=md5_hash, postprocess=gdown.extractall)

    def setup(self, stage: Optional[str] = None):
        self.urban8k_train = self._load_dataset("train", self.train_transforms)
        self.urban8k_val = self._load_dataset("val", self.val_transforms)
        self.urban8k_test = self._load_dataset("test", self.test_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.urban8k_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.urban8k_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            self.urban8k_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        self.urban8k_train = None
        self.urban8k_val = None
        self.urban8k_test = None

    def _load_dataset(self, dataset: str, transforms: Callable) -> DatasetFolder:
        return DatasetFolder(
            os.path.join(self.data_dir, "urban8k_images", dataset),
            loader=lambda src: Image.open(src),
            is_valid_file=lambda src: src.endswith((".jpg", ".jpeg", ".png")),
            transform=transforms,
        )

    def _init_transforms(self) -> T.Compose:
        if self.train_transforms is None:
            self.train_transforms = self.default_transforms
        if self.val_transforms is None:
            self.val_transforms = self.default_transforms
        if self.test_transforms is None:
            self.test_transforms = self.default_transforms

    @property
    def default_transforms(self) -> T.Compose:
        norm_transform = T.Lambda(
            lambda x: torch.log(x + 1e-10) if self.normalize else x
        )
        transforms = T.Compose(
            [
                T.Resize(
                    size=self.image_size, interpolation=T.InterpolationMode.NEAREST,
                ),
                T.ToTensor(),
                norm_transform,
            ]
        )

        return transforms

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--data_dir",
            type=str,
            help="directory containing the train, val and test folders",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="number of samples in a mini batch",
        )
        parser.add_argument(
            "--image_size",
            type=int,
            default=256,
            help="spatial dimensions of the preprocessed image",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="number of worker processes for data loading",
        )
        parser.add_argument(
            "--seed", type=int, default=0, help="seed for reproducibility"
        )
        parser.add_argument(
            "--shuffle",
            type=bool,
            default=True,
            help="whether to shuffle training data",
        )
        parser.add_argument(
            "--pin_memory",
            type=bool,
            default=False,
            help="transfer batches into gpu memory",
        )
        parser.add_argument(
            "--drop_last",
            type=bool,
            default=False,
            help="whether to drop the last incomplete batch",
        )
        parser.add_argument(
            "--normalize",
            type=bool,
            default=True,
            help="whether to normalize the spectrograms",
        )

        return parser

