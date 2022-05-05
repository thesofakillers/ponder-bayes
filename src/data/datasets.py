"""Module for defining Torch Dataset classes"""
from typing import Optional

import torch
from torch.utils.data import Dataset


class ParityDataset(Dataset):
    """torch Dataset for Parity problem described in (Graves, 2016)
    data: binary vectors (values can be 0, 1, -1)
    labels: 1 if odd number of 1's (pos. or neg.) in vector, 0 if even number of 1's

    Credit largely to
    https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/pondernet

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_elems : int
        Size of the vectors.
    seed : int
        seed to use for reproducibility. Use a diff seed for train, val and test
    n_nonzero_min, n_nonzero_max : int or None
        Minimum (inclusive) and maximum (inclusive) number of nonzero
        elements in the feature vector. If not specified then `(1, n_elem)`.
    """

    def __init__(
        self,
        n_samples: int,
        n_elems: int,
        seed: int,
        n_nonzero_min: Optional[int] = None,
        n_nonzero_max: Optional[int] = None,
    ):
        self.n_samples = n_samples
        self.n_elems = n_elems

        self.seed = seed

        self.n_nonzero_min = 1 if n_nonzero_min is None else n_nonzero_min
        self.n_nonzero_max = n_elems if n_nonzero_max is None else n_nonzero_max

        assert (
            0 <= self.n_nonzero_min <= self.n_nonzero_max <= n_elems
        ), "`n_nonzero_min` must be less than or equal to `n_nonzero_max`"

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a feature vector and it's parity (target).
        Note that the generating process is random.
        """
        # use the idx to seed, for reproducibility
        generator = torch.manual_seed(idx + self.seed * self.n_samples)
        # initialize vector of 0s
        x = torch.zeros((self.n_elems,))
        n_non_zero: int = torch.randint(
            low=self.n_nonzero_min,
            high=self.n_nonzero_max + 1,
            size=(1,),
            generator=generator,
        ).item()
        # set the first n_non_zero elements to random values of either 1 or -1
        x[:n_non_zero] = (
            torch.randint(low=0, high=2, size=(n_non_zero,), generator=generator) * 2
            - 1
        )
        # randomly permute the vector so that the non-zero elements are mixed
        x = x[torch.randperm(n=self.n_elems, generator=generator)]

        # generate the label
        y = (x == 1.0).sum() % 2

        return x, y
