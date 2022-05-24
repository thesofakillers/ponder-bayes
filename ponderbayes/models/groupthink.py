import pytorch_lightning as pl

from ponderbayes.models.pondernet import PonderNetModule


class GroupThink(pl.LightningModule):
    """
    PonderNets Together Strong.

    Parameters
    ----------
    n_elems : int
        Number of features in the vector.

    n_hidden : int
        Hidden layer size of the recurrent cell.

    max_steps : int
        Maximum number of steps the network can "ponder" for.

    allow_halting : bool
        If True, then the forward pass is allowed to halt before
        reaching the maximum steps.
    """

    def __init__(
        self,
        ensemble_size=5,
        n_elems=16,
        n_hidden=64,
        max_steps=20,
        allow_halting=False,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.ensemble = []
        self.n_elems = n_elems
        self.n_hidden = n_hidden
        self.max_steps = max_steps
        self.allow_halting = allow_halting
        for _ in range(ensemble_size):
            pondernet = PonderNetModule(n_elems, n_hidden, max_steps, allow_halting)
            self.ensemble.append(pondernet)

    def forward(self, x):

        for pondernet in range(self.ensemble):
            y, p, halting_step = pondernet(x)
