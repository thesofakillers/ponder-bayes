"""
MNIST version
https://github.com/conradkun/PonderNet_MNIST
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from math import floor
import torch.nn.functional as F
from ponderbayes.utils import metrics
from ponderbayes.models import losses

import torchmetrics


class PonderNetMNIST(pl.LightningModule):
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
        n_hidden_cnn=64,
        kernel_size=5,
        max_steps=20,
        allow_halting=False,
        beta=0.01,
        lambda_p=0.4,
    ):
        super().__init__()
        self.n_classes = 10
        self.n_elems = n_elems
        self.n_hidden = n_hidden
        self.max_steps = max_steps
        self.allow_halting = allow_halting
        self.beta = beta
        self.lambda_p = lambda_p

        self.cnn = CNN(n_input=n_elems, kernel_size=kernel_size, n_output=n_hidden_cnn)
        self.cell = nn.GRUCell(n_hidden_cnn, n_hidden)
        self.output_layer = nn.Linear(n_hidden, self.n_classes)
        self.lambda_layer = nn.Linear(n_hidden, 1)

        self.loss_rec_inst = losses.ReconstructionLoss(nn.CrossEntropyLoss()).to(
            torch.float32
        )
        self.loss_reg_inst = losses.RegularizationLoss(
            lambda_p=self.lambda_p, max_steps=self.max_steps
        ).to(torch.float32)

        self.accuracy = torchmetrics.Accuracy()
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
        batch_size = x.shape[0]
        device = x.device

        h = x.new_zeros(batch_size, self.n_hidden)
        embedding = self.cnn(x)
        h = self.cell(embedding, h)

        y_list = []
        p_list = []

        un_halted_prob = x.new_ones(batch_size)
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
            y_list.append(self.output_layer(h))  # (batch_size,n_classes)
            p_list.append(un_halted_prob * lambda_n)  # (batch_size,)

            halting_step = torch.maximum(
                n * (halting_step == 0) * torch.bernoulli(lambda_n).to(torch.long),
                halting_step,
            )

            # Prepare for next iteration
            un_halted_prob = un_halted_prob * (1 - lambda_n)
            embedding = self.cnn(x)
            h = self.cell(embedding, h)

            # Potentially stop if all samples halted
            if self.allow_halting and (halting_step > 0).sum() == batch_size:
                break

        y = torch.stack(y_list)
        p = torch.stack(p_list)

        return y, p, halting_step

    def _accuracy_step(self, y_pred_batch, y_true_batch, halting_step):
        """computes accuracy metrics for a given batch"""

        # (batch_size,) the prediction where the model halted
        # y_halted_batch = y_pred_batch.gather(
        #     dim=0,
        #     index=halting_step[None, :] - 1,
        # )[0]
        # # (scalar), the accuracy at the halted step
        # accuracy_halted_step = metrics.accuracy(
        #     y_halted_batch, y_true_batch, threshold=0
        # )
        # # (max_steps, ), the accuracy at each step
        # accuracy_all_steps = metrics.accuracy(
        #     y_pred_batch, y_true_batch, threshold=0, dim=1
        # )
        batch_size = y_true_batch.shape[0]
        accuracy_all_steps = 0
        halting_step = (
            (halting_step - 1).unsqueeze(0).unsqueeze(2).repeat(1, 1, self.n_classes)
        )

        # calculate the accuracy
        logits = y_pred_batch.gather(dim=0, index=halting_step).squeeze()
        preds = torch.argmax(logits, dim=1)
        accuracy_halted_step = self.accuracy(preds, y_true_batch)

        preds_all_steps = torch.reshape(
            torch.argmax(
                torch.reshape(
                    y_pred_batch, [self.max_steps * batch_size, self.n_classes]
                ),
                dim=1,
            ),
            [self.max_steps, batch_size],
        )
        accuracy_all_steps = torch.Tensor(
            [
                self.accuracy(preds_all_steps[i], y_true_batch)
                for i in range(self.max_steps)
            ]
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
        # y_true_batch = y_true_batch.double()
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


class CNN(nn.Module):
    """
    Simple convolutional neural network.

    Parameters
    ----------
    n_input : int
        Size of the input image. We assume the image is a square,
        and `n_input` is the size of one side.

    n_ouptut : int
        Size of the output.

    kernel_size : int
        Size of the kernel.
    """

    def __init__(self, n_input=28, n_output=50, kernel_size=5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
        self.conv2_drop = nn.Dropout2d()

        # calculate size of convolution output
        self.lin_size = floor(
            (floor((n_input - (kernel_size - 1)) / 2) - (kernel_size - 1)) / 2
        )
        self.fc1 = nn.Linear(self.lin_size**2 * 20, n_output)

    def forward(self, x):
        """forward pass"""
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x
