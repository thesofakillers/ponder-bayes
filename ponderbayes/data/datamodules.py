from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

import ponderbayes.data.datasets as datasets


from torchvision.datasets import MNIST
from torchvision import transforms


class MNIST_DataModule(LightningDataModule):
    """
    DataModule to hold the MNIST dataset.
    Accepts different transforms for train and test to
    allow for extrapolation experiments.

    Parameters
    ----------
    data_dir : str
        Directory where MNIST will be downloaded or taken from.

    train_transform : [transform]
        List of transformations for the training dataset. The same
        transformations are also applied to the validation dataset.

    test_transform : [transform] or [[transform]]
        List of transformations for the test dataset. Also accepts a list of
        lists to validate on multiple datasets with different transforms.

    batch_size : int
        Batch size for both all dataloaders.
    """

    def __init__(
        self,
        data_dir="./data/raw/",
        train_transform=None,
        test_transform=None,
        batch_size=256,
        num_workers=3,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.num_workers = num_workers

        self.default_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        """called only once and on 1 GPU"""
        # download data (train/val and test sets)
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """
        Called on each GPU separately - stage defines if we are
        at fit, validate, test or predict step.
        """
        # we set up only relevant datasets when stage is specified
        if stage in [None, "fit", "validate"]:
            mnist_train = MNIST(
                self.data_dir,
                train=True,
                transform=(self.train_transform or self.default_transform),
            )
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        if stage == "test" or stage is None:
            if self.test_transform is None or isinstance(
                self.test_transform, transforms.Compose
            ):
                self.mnist_test = MNIST(
                    self.data_dir,
                    train=False,
                    transform=(self.test_transform or self.default_transform),
                )
            else:
                self.mnist_test = [
                    MNIST(self.data_dir, train=False, transform=test_transform)
                    for test_transform in self.test_transform
                ]

    def train_dataloader(self):
        """returns training dataloader"""
        mnist_train = DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        return mnist_train

    def val_dataloader(self):
        """returns validation dataloader"""
        mnist_val = DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return mnist_val

    def test_dataloader(self):
        """returns test dataloader(s)"""
        if isinstance(self.mnist_test, MNIST):
            return DataLoader(
                self.mnist_test,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )

        mnist_test = [
            DataLoader(
                test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )
            for test_dataset in self.mnist_test
        ]
        return mnist_test


def get_transforms():
    # define transformations
    transform_22 = transforms.Compose(
        [
            transforms.RandomRotation(degrees=22.5),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    transform_45 = transforms.Compose(
        [
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    transform_67 = transforms.Compose(
        [
            transforms.RandomRotation(degrees=67.5),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    transform_90 = transforms.Compose(
        [
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_transform = transform_22
    test_transform = [transform_22, transform_45, transform_67, transform_90]

    return train_transform, test_transform


class ParityDataModule(LightningDataModule):
    """pytorch lightning DataModule for Parity problem described in (Graves, 2016)
    data: binary vectors (values can be 0, 1, -1)
    labels: 1 if odd number of 1's (pos. or neg.) in vector, 0 if even number of 1's

    Parameters
    ----------
    n_train_samples : int
        Number of training samples to generate.
    n_eval_samples : int
        Number of evaluation samples to generate
    n_elems : int
        Size of the vectors.
        In the case of extrapolation, this defines the
        maximum size of the training vectors, i.e.
        one less than the minimum size of the testing vectors
    mode : str, default "interpolation"
        whether to generate a dataset for interpolation or for extrapolation
    """

    def __init__(
        self,
        n_train_samples,
        n_eval_samples,
        n_elems,
        mode,
        batch_size=64,
        num_workers=3,
    ):
        assert mode in {
            "interpolation",
            "extrapolation",
        }, "`mode` must be one of 'interpolation' or 'extrapolation'"
        assert n_elems % 2 == 0, "`n_elems` must be even"
        self.mode = mode
        self.n_train_samples = n_train_samples
        self.n_eval_samples = n_eval_samples
        self.n_elems = n_elems

        self.batch_size = batch_size
        self.num_workers = num_workers
        super().__init__()

    def prepare_data(self):
        # there is nothing to download or save
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.parity_train = datasets.ParityDataset(
                n_samples=self.n_train_samples,
                n_elems=self.n_elems,
                mode=self.mode,
                split="train",
            )
            self.parity_val = datasets.ParityDataset(
                n_samples=self.n_eval_samples,
                n_elems=self.n_elems,
                mode=self.mode,
                split="val",
            )
        if stage in (None, "test"):
            self.parity_test = datasets.ParityDataset(
                n_samples=self.n_eval_samples,
                n_elems=self.n_elems,
                mode=self.mode,
                split="test",
            )
        if stage in (None, "debug"):
            self.parity_debug = datasets.ParityDataset(
                n_samples=200,
                n_elems=16,
                mode=self.mode,
                split="val",
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.parity_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.parity_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.parity_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def debug_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.parity_debug,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
