"""Module for defining Torch Dataset classes"""
from typing import Optional

import torch
from torch.utils.data import Dataset


class ParityDataset(Dataset):
    """torch Dataset for Parity problem described in (Graves, 2016)
    data: binary vectors (values can be 0, 1, -1)
    labels: 1 if odd number of 1's (pos. or neg.) in vector, 0 if even number of 1's

    Credit in part to
    https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/pondernet

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_elems : int
        Size of the vectors.
    mode : str, default "interpolation"
        whether to generate a dataset for interpolation or for extrapolation
    split : str, default "train"
        whether to generate the train, validation or test split
    n_nonzero_min, n_nonzero_max : int or None
        Minimum (inclusive) and maximum (inclusive) number of nonzero
        elements in the training feature vectors.
        If not specified then `(1, n_elem)`.
    """

    def __init__(
        self,
        n_samples: int,
        n_elems: int,
        mode: str = "interpolation",
        split: str = "train",
        n_nonzero_min: Optional[int] = None,
        n_nonzero_max: Optional[int] = None,
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
        self.n_samples = n_samples
        self.n_elems = n_elems

        self.n_nonzero_min = 1 if n_nonzero_min is None else n_nonzero_min
        self.n_nonzero_max = n_elems if n_nonzero_max is None else n_nonzero_max

        assert (
            0 <= self.n_nonzero_min <= self.n_nonzero_max <= n_elems
        ), "`n_nonzero_min` must be less than or equal to `n_nonzero_max`"

    def __len__(self) -> int:
        return self.n_samples

    def _generate_parity_sample(self, x, generator, idx):
        """
        x : torch.Tensor
            vector of 0s, of any length
        """
        # consistently sample a different difficulty
        n_non_zero = idx % self.n_nonzero_max
        # set the first n_non_zero elements to random values of either 1 or -1
        output = x.clone()
        output[:n_non_zero] = (
            torch.randint(low=0, high=2, size=(n_non_zero,), generator=generator) * 2
            - 1
        )
        # randomly permute the vector so that the non-zero elements are mixed
        output = output[torch.randperm(n=self.n_elems, generator=generator)]
        # generate the label
        y = (torch.abs(output) == 1.0).sum() % 2
        return output, y

    def _get_interpolation_item(self, generator, idx):
        # initialize vector of 0s
        x = torch.zeros((self.n_elems,))
        return self._generate_parity_sample(x, generator, idx)

    def _get_extrapolation_item(self, generator):
        if self.split == "train":
            # need a length of at least n_nonzero_max, at most n_elems
            vector_length = torch.randint(
                low=self.n_nonzero_max,
                high=self.n_elems + 1,
                size=(1,),
                generator=generator,
            ).item()
        else:
            vector_length = torch.randint(
                low=self.n_elems + 1,
                high=self.n_elems * 2 + 1,
                size=(1,),
                generator=generator,
            ).item()
        x = torch.zeros((vector_length,))
        return self._generate_parity_sample(x, generator)

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
        if self.mode == "interpolation":
            return self._get_interpolation_item(generator, idx)
        elif self.mode == "extrapolation":
            return self._get_extrapolation_item(generator)
