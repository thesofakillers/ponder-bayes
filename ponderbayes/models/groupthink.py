import torch

from ponderbayes.models.pondernet import PonderNetModule, PonderNet


class GroupThink(PonderNet):
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
        beta=0.01,
        lambda_p=0.4,
    ):
        super().__init__(n_elems, n_hidden, max_steps, allow_halting, beta, lambda_p)
        self.ensemble_size = ensemble_size
        self.ensemble = []
        for _ in range(ensemble_size):
            pondernet = PonderNetModule(n_elems, n_hidden, max_steps, allow_halting)
            self.ensemble.append(pondernet)

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.Tensor
            Batch of input features of shape `(batch_size, n_elems)`.
        """
        batch_size = x.shape[0]
        y_ensemble = torch.zeros(self.ensemble_size, self.max_steps, batch_size)
        p_ensemble = torch.zeros(self.ensemble_size, self.max_steps, batch_size)
        halting_step_ensemble = torch.zeros(self.ensemble_size, batch_size)

        for i, pondernet in enumerate(self.ensemble):
            y, p, halting_step = pondernet(x)
            y_ensemble[i] = y
            p_ensemble[i] = p
            halting_step_ensemble[i] = halting_step

        return y_ensemble, p_ensemble, halting_step_ensemble

    def _shared_step(self, batch, batch_idx, phase):
        """
        Runs forward, computes accuracy and loss and logs

        The y_preds, p and halting_steps are averaged across the ensemble
        """
        x, y_true = batch
        y_true = y_true.double()

        # same as tensors below, but additional first dimension of ensemble_size
        y_pred_ensemble, p_ensemble, halting_step_ensemble = self(x)

        # (max_steps, batch_size)
        y_pred_mean = torch.mean(y_pred_ensemble, dim=0)
        y_pred_std = torch.std(y_pred_ensemble, dim=0)

        # (max_steps, batch_size)
        std_err = y_pred_std / torch.sqrt(torch.Tensor([self.ensemble_size]))

        # (max_steps, batch_size)
        p_mean = torch.mean(p_ensemble, dim=0)
        p_std = torch.std(p_ensemble, dim=0)

        # (batch_size, )
        # TODO: don't do this,
        # compute the accuracy for each model at the relevant halting step
        # and average that
        # same for std_err
        halting_step_mean = torch.mean(halting_step_ensemble, dim=0).floor().long()
        halting_step_std = torch.std(halting_step_ensemble, dim=0)

        # index std_err by halting step mean, to get (batch_size, ) tensor
        std_err_halting_step = std_err.gather(
            dim=0, index=halting_step_mean[None, :] - 1
        ).squeeze()

        accuracy_halted_step, accuracy_all_steps = self._accuracy_step(
            y_pred_mean, y_true, halting_step_mean
        )
        # currently taking the loss of the mean
        # if that doesn't work, try taking the loss of each element in the ensemble
        loss_rec, loss_reg, loss_overall = self._loss_step(p_mean, y_pred_mean, y_true)

        # collating all results
        results = {
            "halting_step": halting_step_mean.double().mean(),
            "p": p_mean.mean(dim=1),
            "accuracy_halted_step": accuracy_halted_step,
            "accuracy_all_steps": accuracy_all_steps,
            "loss_rec": loss_rec,
            "loss_reg": loss_reg,
            "loss": loss_overall,
            "std_err": std_err.mean(dim=1),
            "std_err_halting_step": std_err_halting_step.mean(),
        }
        # logging; p and accuracy_all_steps logged in _shared_epoch_end
        self.log_dict(
            {
                f"{phase}/{k}": results[k]
                for k in results.keys()
                if k not in ["accuracy_all_steps", "p", "std_err"]
            }
        )
        # needed for backward
        return results

    # def _shared_epoch_end(self, outputs, phase):
    #     """
    #     Accumulates and logs per-step metrics at the end of the epoch

    #     Each metric is averaged across the batches
    #     """
    #     accuracy_all_steps = torch.stack(
    #         [output["accuracy_all_steps"] for output in outputs]
    #     ).mean(dim=0)
    #     p = torch.stack([output["p"] for output in outputs]).mean(dim=0)
    #     std_err = torch.stack([output["std_err"] for output in outputs]).mean(dim=0)
    #     for i, (accuracy, step_p, step_std_err) in enumerate(
    #         zip(accuracy_all_steps, p, std_err), start=1
    #     ):
    #         self.log(f"{phase}/step_accuracy/{i}", accuracy)
    #         self.log(f"{phase}/step_p/{i}", step_p)
    #         self.log(f"{phase}/step_std_err/{i}", step_std_err)
