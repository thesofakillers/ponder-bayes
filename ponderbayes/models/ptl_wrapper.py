"""
Credit for PonderNet, ReconstructionLoss and RegularizationLoss largely to
https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/pondernet
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from torch.distributions import constraints
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.nn.module import to_pyro_module_
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal, AutoMultivariateNormal, init_to_mean
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDiagonalNormal

from ponderbayes.utils import metrics
from ponderbayes.models import losses, ponderbayes


class PtlWrapper(pl.LightningModule):
    """Network that ponders.
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
    Attributes
    ----------
    cell : nn.GRUCell
        Learnable GRU cell that maps the previous hidden state and the input
        to a new hidden state.
    output_layer : nn.Linear
        Linear module that serves as the binary classifier. It inputs
        the hidden state.
    lambda_layer : nn.Linear
        Linear module that generates the halting probability at each step.
    """

    def __init__(
        self,
        n_elems,
        n_hidden=64,
        max_steps=20,
        allow_halting=False,
        beta=0.01,
        lambda_p=0.4,
        lr=0.0003,
    ):
        super().__init__()

        self.n_elems = n_elems
        self.n_hidden = n_hidden
        self.max_steps = max_steps
        self.allow_halting = allow_halting
        self.beta = beta
        self.lambda_p = lambda_p
        self.lr = lr

        self.net = ponderbayes.PonderBayes(
            self.n_elems,
            self.n_hidden,
            self.max_steps,
            self.allow_halting,
            self.beta,
            self.lambda_p,
        )

        # necessary for Bayesian Inference
        self.automatic_optimization = False
        self.guide = AutoDiagonalNormal(self.net)

        self.loss_reg_inst = losses.RegularizationLoss(
            lambda_p=self.lambda_p, max_steps=self.max_steps
        ).to(torch.float32)

        self.save_hyperparameters()

    def on_train_start(self):
        pyro.clear_param_store()
        opt = pyro.optim.ClippedAdam({"lr": self.lr, "clip_norm": 1.0})
        self.svi = SVI(self.net, self.guide, opt, loss=losses.custom_loss)

    def forward(self, x, y_true=None):
        return self.net(x, y_true)

    # TODO: needs fixing (victor's code?)
    def _accuracy_step(self, y_pred_batch, y_true_batch, halting_step):
        """computes accuracy metrics for a given batch"""
        # (batch_size,) the prediction where the model halted
        y_halted_batch = y_pred_batch.gather(
            dim=0,
            index=halting_step[None, :] - 1,
        )[0]
        # (scalar), the accuracy at the halted step
        accuracy_halted_step = metrics.accuracy(
            y_halted_batch, y_true_batch, threshold=0
        )
        # (max_steps, ), the accuracy at each step
        accuracy_all_steps = metrics.accuracy(
            y_pred_batch, y_true_batch, threshold=0, dim=1
        )
        return accuracy_halted_step, accuracy_all_steps

    def training_step(self, batch, batch_idx):
        """runs forward, computes accuracy and loss and logs"""
        # (batch_size, n_elems), (batch_size,)
        x, y_true_batch = batch
        y_true = y_true_batch.double()

        loss = self.svi.step(x, y_true)
        print("did forward")
        # , y_pred_batch, p, halting_step
        print(loss)

        # TODO: needs fixing (victor's code?)

        # # (max_steps, batch_size), (max_steps, batch_size), (batch_size,)
        y_pred_batch, p, halting_step = self(batch)

        # batch accuracy at the halted step, batch accuracy at each step
        accuracy_halted_step, accuracy_all_steps = self._accuracy_step(
            y_pred_batch, y_true_batch, halting_step
        )

        # # reconstruction, regularization and overall loss
        # loss_rec, loss_reg, loss_overall = self._loss_step(
        #     p, y_pred_batch, y_true_batch
        # )

        # collating all results
        results = {
            "halting_step": halting_step.double().mean(),
            "p": p.mean(dim=1),
            "accuracy_halted_step": accuracy_halted_step,
            "accuracy_all_steps": accuracy_all_steps,
            "loss": loss,
        }
        # logging; p and accuracy_all_steps logged in _shared_epoch_end
        self.log_dict(
            {
                f"{phase}/{k}": results[k]
                for k in [
                    "loss",
                    "halting_step",
                    "accuracy_halted_step",
                ]
            }
        )
        # needed for backward
        return results

    def validation_step(self, batch, batch_idx):
        # TODO: figure this val step (victor's code?)
        return torch.rand(len(batch))

    def test_step(self, batch, batch_idx):
        # TODO: figure this test step (victor's code?)
        return torch.rand(len(batch))

    # TODO: needs fixing (victor's code?)
    def _shared_epoch_end(self, outputs, phase):
        """Accumulates and logs per-step metrics at the end of the epoch"""
        accuracy_all_steps = torch.stack(
            [output["accuracy_all_steps"] for output in outputs]
        ).mean(dim=0)
        p = torch.stack([output["p"] for output in outputs]).mean(dim=0)
        for i, (accuracy, step_p) in enumerate(zip(accuracy_all_steps, p), start=1):
            self.log(f"{phase}/step_accuracy/{i}", accuracy)
            self.log(f"{phase}/step_p/{i}", step_p)

    def training_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        """Handles optimizers and schedulers"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
