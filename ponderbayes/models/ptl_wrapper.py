"""
Credit for PonderNet, ReconstructionLoss and RegularizationLoss largely to
https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/pondernet
"""
from trace import Trace
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from torch.distributions import constraints
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.nn.module import to_pyro_module_
from pyro.infer import SVI, TraceMeanField_ELBO, Predictive
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
        num_samples=5,
    ):
        super().__init__()

        self.n_elems = n_elems
        self.n_hidden = n_hidden
        self.max_steps = max_steps
        self.allow_halting = allow_halting
        self.beta = beta
        self.lambda_p = lambda_p
        self.lr = lr
        self.num_samples = num_samples

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
        self.guide = AutoMultivariateNormal(self.net, init_loc_fn=init_to_mean)
        # self.guide = ponderbayes.MyGuide(self.net)

        self.save_hyperparameters()

    def on_train_start(self):
        pyro.clear_param_store()
        opt = pyro.optim.Adam(
            {"lr": self.lr, "betas": (0.90, 0.999)}, {"clip_norm": 1.0}
        )
        self.svi = SVI(
            self.net.to(self.device),
            self.guide.to(self.device),
            opt,
            loss=losses.custom_loss,
            # TraceMeanField_ELBO(),
        )

    def forward(self, x, y_true=None):
        return self.net(x, y_true)

    def _shared_step(self, x_batch, y_true_batch):
        # Define what we want to be returned from the predictive
        device = x_batch.device
        predictions = self.net.predict(self.guide, self.num_samples, x_batch)

        # Separate p which is the probabilities of all steps and the
        # halting step which is the step at which the model halted
        p, halting_step = torch.split(predictions["_RETURN"], self.max_steps, dim=1)
        p = p.mean(dim=0)
        halting_step = halting_step.to(int).squeeze()

        # From the observations collect the prediction of the model at the halting step
        y_pred_batch = torch.zeros([self.num_samples, self.max_steps, x_batch.shape[0]])
        for obs_n in range(self.net.max_steps):
            y_pred_batch[:, obs_n, :] = predictions[f"obs_{obs_n}"]

        y_pred_batch = y_pred_batch.to(device)
        accuracy_halted_step = torch.zeros([self.num_samples])
        accuracy_all_steps = torch.zeros([self.num_samples, self.max_steps])

        for i in range(self.num_samples):
            (
                accuracy_halted_step[i],
                accuracy_all_steps[i],
            ) = self._accuracy_sample(y_pred_batch[i], y_true_batch, halting_step[i])
        results = {
            "halting_step": halting_step.double().mean(),
            "p": p.mean(dim=1),
            "accuracy_halted_step": accuracy_halted_step.mean(dim=0),
            "accuracy_all_steps": accuracy_all_steps.mean(dim=0),
            "accuracy_halted_step_std": accuracy_halted_step.std(dim=0),
            "accuracy_all_steps_std": accuracy_all_steps.std(dim=0),
        }
        return results

    # TODO: needs fixing (victor's code?)
    def _accuracy_sample(self, y_pred_batch, y_true_batch, halting_step):
        """computes accuracy metrics for a given batch"""
        # (batch_size,) the prediction where the model halted
        y_halted_batch = y_pred_batch.gather(
            dim=0,
            index=halting_step[None, :] - 1,
        )[0]
        # (scalar), the accuracy at the halted step
        accuracy_halted_step = metrics.accuracy(
            y_halted_batch,
            y_true_batch,
        )
        # (max_steps, ), the accuracy at each step
        accuracy_all_steps = metrics.accuracy(y_pred_batch, y_true_batch, dim=1)
        return accuracy_halted_step, accuracy_all_steps

    def training_step(self, batch, batch_idx):
        """runs forward, computes accuracy and loss and logs"""
        # (batch_size, n_elems), (batch_size,)
        x_batch, y_true = batch
        y_true = y_true.double()

        loss = torch.Tensor([self.svi.step(x_batch, y_true)])
        results = self._shared_step(x_batch, y_true)

        results["loss"] = loss

        self.log_dict(
            {
                f"train/{k}": results[k]
                for k in [
                    "loss",
                    "halting_step",
                    "accuracy_halted_step",
                    "accuracy_halted_step_std",
                ]
            },
            prog_bar=True,
        )
        return results

    def _eval_step(self, batch, phase):
        x_batch, y_true_batch = batch
        y_true_batch = y_true_batch.double()

        loss = self.svi.evaluate_loss(x_batch, y_true_batch)

        results = self._shared_step(x_batch, y_true_batch)

        results["loss"] = loss

        self.log_dict(
            {
                f"{phase}/{k}": results[k]
                for k in [
                    "loss",
                    "halting_step",
                    "accuracy_halted_step",
                    "accuracy_halted_step_std",
                ]
            }
        )
        return results

    def validation_step(self, batch, batch_idx):
        # TODO: figure this val step (victor's code?)
        return self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        # TODO: figure this test step (victor's code?)
        return self._eval_step(batch, "test")

    # TODO: needs fixing (victor's code?)
    def _shared_epoch_end(self, outputs, phase):
        """Accumulates and logs per-step metrics at the end of the epoch"""
        accuracy_all_steps = torch.stack(
            [output["accuracy_all_steps"] for output in outputs]
        ).mean(dim=0)
        accuracy_all_steps_std = torch.stack(
            [output["accuracy_all_steps_std"] for output in outputs]
        ).mean(dim=0)
        p = torch.stack([output["p"] for output in outputs]).mean(dim=0)
        for i, (accuracy, accuracy_std, step_p) in enumerate(
            zip(accuracy_all_steps, accuracy_all_steps_std, p), start=1
        ):
            self.log(f"{phase}/step_accuracy/{i}", accuracy)
            self.log(f"{phase}/step_accuracy_std/{i}", accuracy_std)
            self.log(f"{phase}/step_p/{i}", step_p)

    def training_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        """Handles optimizers and schedulers.
        This optimizer is never used"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
