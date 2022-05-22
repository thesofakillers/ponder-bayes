from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import ponderbayes.data.datasets as datasets


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
