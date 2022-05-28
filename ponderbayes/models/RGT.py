import torch
import torch.nn as nn
import pytorch_lightning as pl

from ponderbayes.models.pondernet import PonderNetModule
from ponderbayes.models import losses
from ponderbayes import utils


class RationalGroupThink(pl.LightningModule):
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
    beta : float
        Hyperparameter governing the regularization loss
    lambda_p : float
        Hyperparameter governing the probability of halting
    """

    def __init__(
        self,
        ensemble_size=5,
        n_elems=16,
        n_hidden=64,
        max_steps=20,
        allow_halting=False,
        beta=0.01,
    ):
        super().__init__()

        self.n_elems = n_elems
        self.n_hidden = n_hidden
        self.max_steps = max_steps
        self.allow_halting = allow_halting
        self.beta = beta

        self.ensemble_size = ensemble_size
        self.ensemble = nn.ModuleList(
            [
                PonderNetModule(n_elems, n_hidden, max_steps, allow_halting)
                for _ in range(ensemble_size)
            ]
        )
        # can't rely on automatic optimization from pl here
        self.automatic_optimization = False

        self.loss_rec_inst = losses.ReconstructionLoss(
            nn.BCEWithLogitsLoss(reduction="none")
        ).to(torch.float32)
        self.loss_reg_inst = losses.DynamicRegularizationLoss(
            max_steps=self.max_steps
        ).to(torch.float32)

        self.save_hyperparameters()

        # initialize std error to maximum
        self.std_err_halted = (0.5 / torch.sqrt(torch.Tensor([self.ensemble_size])))[
            :, None
        ].detach()

        self.lambda_p_mlp = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

    def forward(self, x):
        """
        Forward pass, convenient for inference, but not actually used for
        training

        Parameters
        ----------
        x : torch.Tensor
            Batch of input features of shape `(batch_size, n_elems)`.
        """
        y_list = []
        p_list = []
        halting_step_list = []

        for i in range(self.ensemble_size):
            y, p, halting_step = self.ensemble[i](x)
            y_list.append(y)
            p_list.append(p)
            halting_step_list.append(halting_step)

        # stack each list into single tensor (ensemble_size, max_steps, batch_size)
        y_ensemble = torch.stack(y_list, dim=0)
        p_ensemble = torch.stack(p_list, dim=0)
        halting_step_ensemble = torch.stack(halting_step_list, dim=0)

        return y_ensemble, p_ensemble, halting_step_ensemble

    def _shared_step(self, batch, batch_idx, phase):
        """
        Runs forward, computes accuracy and loss and logs

        The y_preds, p and halting_steps are averaged across the ensemble
        """
        x, y_true = batch
        batch_size = x.shape[0]
        y_true = y_true.double()
        if batch_idx == 0 and self.current_epoch == 0:
            # we don't know the batch size upon initialization
            self.std_err_halted = self.std_err_halted.expand(batch_size, 1).detach()
        # compute lambda_p (batch_size,) with uncertainty from previous batch
        lambda_p = self.lambda_p_mlp(self.std_err_halted).squeeze()

        # accuracy at the halting step for each model
        accuracies = torch.zeros(self.ensemble_size, batch_size, requires_grad=False)

        # loss for each model
        losses_rec = torch.zeros(self.ensemble_size, requires_grad=False)
        losses_reg = torch.zeros(self.ensemble_size, requires_grad=False)

        # (sigmoid-scaled) y_pred at halting step for each model (batch_size, )
        y_pred_halted_ensemble = torch.zeros(self.ensemble_size, batch_size)
        # halting step of each model
        halting_step_ensemble = torch.zeros(self.ensemble_size, batch_size)

        # perform forward and evaluate each model
        for i, model in enumerate(self.ensemble):
            # forward pass
            y_pred, p, halting_step = model(x)
            y_pred_halted = utils.tensor_at_halting_step(y_pred, halting_step.long())
            # accuracy
            accuracy = self._accuracy_step(y_pred_halted, y_true)
            accuracies[i] = accuracy
            # apply sigmoid to normalize our output for valid std_err calculation
            with torch.no_grad():
                y_pred_halted_scaled = torch.sigmoid(y_pred_halted).detach()
                y_pred_halted_ensemble[i] = y_pred_halted_scaled
                halting_step_ensemble[i] = halting_step
            # losses
            if phase == "train":
                opt = self.optimizers()[i]
                opt.zero_grad()
            loss_rec, loss_reg, loss_overall = self._loss_step(
                p, y_pred, y_true, lambda_p
            )
            # optimization per model
            retain_graph = True if i < self.ensemble_size - 1 else False
            if phase == "train":
                # loss_overall.backward(retain_graph=retain_graph)
                self.manual_backward(loss_overall, retain_graph=retain_graph)
                opt.step()
            losses_rec[i] = loss_rec.detach()
            losses_reg[i] = loss_reg.detach()

        losses_overall = losses_rec + self.beta * losses_reg

        # get mean accuracy
        accuracy_halted_step = torch.mean(accuracies, dim=0)
        # get std err of (sigmoid-scaled) prediction
        std_dev_halted = torch.std(y_pred_halted_ensemble, dim=0).detach()
        # (batch_size, 1)
        self.std_err_halted = (
            std_dev_halted / torch.sqrt(torch.Tensor([self.ensemble_size]))
        )[:, None].detach()

        # collating all results
        results = {
            "halting_step": halting_step_ensemble.double().mean(),
            "accuracy_halted_step": accuracy_halted_step.mean(),
            "accuracy_per_model": accuracies.mean(dim=1),
            "loss_rec": losses_rec.mean(dim=0),
            "loss_reg": losses_reg.mean(dim=0),
            "loss": losses_overall.mean(dim=0),
            "std_err": self.std_err_halted.mean(),
            "lambda_p": lambda_p.detach().mean(),
        }
        # logging; accuracy_per_model logged in _shared_epoch_end
        self.log_dict(
            {
                f"{phase}/{k}": v
                for k, v in results.items()
                if k not in ["accuracy_per_model"]
            }
        )
        return results

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def _accuracy_step(self, y_pred_halted, y_true):
        accuracy_halted_step = utils.metrics.accuracy(
            y_pred_halted, y_true, threshold=0
        )
        return accuracy_halted_step

    def _loss_step(self, p, y_pred_batch, y_true_batch, lambda_p):
        """computes the loss for a given batch"""
        # reconstruction loss
        loss_rec = self.loss_rec_inst(p, y_pred_batch, y_true_batch)
        # regularization loss
        loss_reg = self.loss_reg_inst(p, lambda_p)
        # overall loss
        loss_overall = loss_rec + self.beta * loss_reg
        return loss_rec, loss_reg, loss_overall

    def _shared_epoch_end(self, outputs, phase):
        """Accumulates and logs per-model metrics at the end of the epoch"""
        # (n_batches, n_models)
        accuracy_per_model = torch.stack(
            [output["accuracy_per_model"] for output in outputs]
        ).mean(dim=0)
        for i, accuracy in enumerate(accuracy_per_model, start=1):
            self.log(f"{phase}/model_{i}/accuracy", accuracy)

    def training_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizers = []
        for model in self.ensemble:
            # https://discuss.pytorch.org/t/adam-half-precision-nans/1765
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, eps=1e-4)
            optimizers.append(optimizer)
        return optimizers

    def configure_gradient_clipping(self, optimizer, optimizer_idx):
        self.clip_gradients(optimizer, 1, "norm")
