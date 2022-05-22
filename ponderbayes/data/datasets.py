"""Module for defining Torch Dataset classes"""
import torch
from torch.utils.data import Dataset


class ParityDataset(Dataset):
    """torch Dataset for Parity problem described in (Graves, 2016)
    data: binary vectors (values can be 0, 1, -1)
    labels: 1 if odd number of 1's (pos. or neg.) in vector, 0 if even number of 1's

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_elems : int
        Size of the vectors and also the maximum difficulty of the validation set.
    mode : str, default "interpolation"
        whether to generate a dataset for interpolation or for extrapolation
    split : str, default "train"
        whether to generate the train, validation or test split
    """

    def __init__(
        self,
        n_samples: int,
        n_elems: int,
        mode: str,
        split: str,
    ):
        assert mode in {
            "interpolation",
            "extrapolation",
        }, "`mode` must be one of 'interpolation' or 'extrapolation'"
        assert split in {
            "train",
            "val",
            "test",
        }, "`split` must be one of 'train', 'val' or 'test'"
        assert n_elems % 2 == 0, "`n_elems` must be even"
        self.mode = mode
        self.split = split
        self.n_samples = int(n_samples)
        self.n_elems = int(n_elems)
        # prepare possible amounts of nonzeros ahead of time
        if self.mode == "extrapolation":
            if self.split != "train":
                self.possible_nonzeros = range(self.n_elems // 2 + 1, self.n_elems + 1)
            elif self.split == "train":
                self.possible_nonzeros = range(1, self.n_elems // 2 + 1)
        else:
            self.possible_nonzeros = range(1, self.n_elems + 1)

    def __len__(self) -> int:
        return self.n_samples

    def _generate_parity_sample(self, generator, cur_idx):
        """Generates a parity sample"""
        # initialize the feature vector
        x = torch.zeros((self.n_elems,))
        # ensuring a uniform distribution of difficulty across dataset
        if self.mode == "extrapolation":
            modulo_idx = int(cur_idx % (self.n_elems // 2))
        else:
            modulo_idx = int(cur_idx % self.n_elems)
        n_nonzero = self.possible_nonzeros[modulo_idx]
        # set the first n_non_zero elements to random values of either 1 or -1
        output = x.clone()
        output[:n_nonzero] = (
            torch.randint(low=0, high=2, size=(n_nonzero,), generator=generator) * 2 - 1
        )
        # randomly permute the vector so that the non-zero elements are mixed
        output = output[torch.randperm(n=self.n_elems, generator=generator)]
        # generate the label
        y = (output != 0.0).sum() % 2
        return output, y

    def __getitem__(self, idx):
        """
        Get a feature vector and it's parity (target).
        Note that the generating process is random.
        """
        if idx >= self.n_samples:
            raise IndexError()
        # we use this to avoid overlap between train/val/test in interpolation
        split_to_multiplier = {"train": 1, "val": 2, "test": 3}
        # use the idx to seed, for reproducibility
        generator = torch.manual_seed(
            idx + split_to_multiplier[self.split] * self.n_samples
        )
        return self._generate_parity_sample(generator, idx)
