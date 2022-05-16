"""
Credit for PonderNet, ReconstructionLoss and RegularizationLoss in part to
https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/pondernet
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn

from ponderbayes.utils import metrics


class PonderNet(pl.LightningModule):
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
    beta : float
        Hyperparameter governing the regularization loss
    lambda_p : float
        Hyperparameter governing the probability of halting
    """

    def __init__(
        self,
        n_elems,
        n_hidden=64,
        max_steps=20,
        allow_halting=False,
        beta=0.01,
        lambda_p=0.4,
    ):
        super().__init__()

        self.n_elems = n_elems
        self.n_hidden = n_hidden
        self.max_steps = max_steps
        self.allow_halting = allow_halting
        self.beta = beta
        self.lambda_p = lambda_p

        self.cell = nn.GRUCell(n_elems, n_hidden)
        self.output_layer = nn.Linear(n_hidden, 1)
        self.lambda_layer = nn.Linear(n_hidden, 1)

        self.loss_rec_inst = ReconstructionLoss(
            nn.BCEWithLogitsLoss(reduction="none")
        ).to(torch.float32)
        self.loss_reg_inst = RegularizationLoss(
            lambda_p=self.lambda_p, max_steps=self.max_steps
        ).to(torch.float32)

        self.save_hyperparameters()

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input features of shape `(batch_size, n_elems)`.

        Returns
        -------
        y : torch.Tensor
            Tensor of shape `(max_steps, batch_size)` representing
            the predictions for each step and each sample. In case
            `allow_halting=True` then the shape is
            `(steps, batch_size)` where `1 <= steps <= max_steps`.

        p : torch.Tensor
            Tensor of shape `(max_steps, batch_size)` representing
            the halting probabilities. Sums over rows (fixing a sample)
            are 1. In case `allow_halting=True` then the shape is
            `(steps, batch_size)` where `1 <= steps <= max_steps`.

        halting_step : torch.Tensor
            An integer for each sample in the batch that corresponds to
            the step when it was halted. The shape is `(batch_size,)`. The
            minimal value is 1 because we always run at least one step.
        """
        batch_size, _ = x.shape
        device = x.device

        h = x.new_zeros(batch_size, self.n_hidden)

        un_halted_prob = x.new_ones(batch_size)

        y_list = []
        p_list = []

        halting_step = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=device,
        )

        for n in range(1, self.max_steps + 1):
            if n == self.max_steps:
                lambda_n = x.new_ones(batch_size)  # (batch_size,)
            else:
                lambda_n = torch.sigmoid(self.lambda_layer(h))[:, 0]  # (batch_size,)

            # Store releavant outputs
            y_list.append(self.output_layer(h)[:, 0])  # (batch_size,)
            p_list.append(un_halted_prob * lambda_n)  # (batch_size,)

            halting_step = torch.maximum(
                n * (halting_step == 0) * torch.bernoulli(lambda_n).to(torch.long),
                halting_step,
            )

            # Prepare for next iteration
            un_halted_prob = un_halted_prob * (1 - lambda_n)
            h = self.cell(x, h)

            # Potentially stop if all samples halted
            if self.allow_halting and (halting_step > 0).sum() == batch_size:
                break

        y = torch.stack(y_list)
        p = torch.stack(p_list)

        return y, p, halting_step

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

    def _loss_step(self, p, y_pred_batch, y_true_batch):
        """computes the loss for a given batch"""
        # reconstruction loss
        loss_rec = self.loss_rec_inst(
            p,
            y_pred_batch,
            y_true_batch,
        )
        # regularization loss
        loss_reg = self.loss_reg_inst(
            p,
        )
        # overall loss
        loss_overall = loss_rec + self.beta * loss_reg
        return loss_rec, loss_reg, loss_overall

    def _shared_step(self, batch, batch_idx, phase):
        """runs forward, computes accuracy and loss and logs"""
        # (batch_size, n_elems), (batch_size,)
        x_batch, y_true_batch = batch
        y_true_batch = y_true_batch.double()
        # (max_steps, batch_size), (max_steps, batch_size), (batch_size,)
        y_pred_batch, p, halting_step = self(x_batch)
        # batch accuracy at the halted step, batch accuracy at each step
        accuracy_halted_step, accuracy_all_steps = self._accuracy_step(
            y_pred_batch, y_true_batch, halting_step
        )
        # reconstruction, regularization and overall loss
        loss_rec, loss_reg, loss_overall = self._loss_step(
            p, y_pred_batch, y_true_batch
        )
        # collating all results
        results = {
            "halting_step": halting_step.double().mean(),
            "p": p.mean(dim=1),
            "accuracy_halted_step": accuracy_halted_step,
            "accuracy_all_steps": accuracy_all_steps,
            "loss_rec": loss_rec,
            "loss_reg": loss_reg,
            "loss": loss_overall,
        }
        # logging; p and accuracy_all_steps logged in _shared_epoch_end
        self.log_dict(
            {
                f"{phase}/{k}": results[k]
                for k in [
                    "loss_rec",
                    "loss_reg",
                    "loss",
                    "halting_step",
                    "accuracy_halted_step",
                ]
            }
        )
        # needed for backward
        return results

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)
        return optimizer


class ReconstructionLoss(nn.Module):
    """Weighted average of per step losses.

    Parameters
    ----------
    loss_func : callable
        Loss function that accepts `y_pred` and `y_true` as arguments. Both
        of these tensors have shape `(batch_size,)`. It outputs a loss for
        each sample in the batch.
    """

    def __init__(self, loss_func):
        super().__init__()

        self.loss_func = loss_func

    def forward(self, p, y_pred, y_true):
        """Compute loss.

        Parameters
        ----------
        p : torch.Tensor
            Probability of halting of shape `(max_steps, batch_size)`.

        y_pred : torch.Tensor
            Predicted outputs of shape `(max_steps, batch_size)`.

        y_true : torch.Tensor
            True targets of shape `(batch_size,)`.

        Returns
        -------
        loss : torch.Tensor
            Scalar representing the reconstruction loss. It is nothing else
            than a weighted sum of per step losses.
        """
        max_steps, _ = p.shape
        total_loss = p.new_tensor(0.0)

        for n in range(max_steps):
            loss_per_sample = p[n] * self.loss_func(y_pred[n], y_true)  # (batch_size,)
            total_loss = total_loss + loss_per_sample.mean()  # (1,)

        return total_loss


class RegularizationLoss(nn.Module):
    """Enforce halting distribution to ressemble the geometric distribution.

    Parameters
    ----------
    lambda_p : float
        The single parameter determining uniquely the geometric distribution.
        Note that the expected value of this distribution is going to be
        `1 / lambda_p`.

    max_steps : int
        Maximum number of pondering steps.
    """

    def __init__(self, lambda_p, max_steps=20):
        super().__init__()

        p_g = torch.zeros((max_steps,))
        not_halted = 1.0

        for k in range(max_steps):
            p_g[k] = not_halted * lambda_p
            not_halted = not_halted * (1 - lambda_p)

        self.register_buffer("p_g", p_g)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, p):
        """Compute loss.

        Parameters
        ----------
        p : torch.Tensor
            Probability of halting of shape `(steps, batch_size)`.

        Returns
        -------
        loss : torch.Tensor
            Scalar representing the regularization loss.
        """
        steps, batch_size = p.shape

        p = p.transpose(0, 1)  # (batch_size, max_steps)

        p_g_batch = self.p_g[None, :steps].expand_as(p)  # (batch_size, max_steps)

        return self.kl_div(p.log(), p_g_batch)
