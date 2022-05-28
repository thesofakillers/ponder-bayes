import torch
import torch.nn as nn


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


class DynamicRegularizationLoss(nn.Module):
    """
    Enforce halting distribution to resemble
    a dynamically changing geometric distribution.

    Parameters
    ----------
    max_steps : int
        Maximum number of pondering steps.
    """

    def __init__(self, max_steps=20):
        super().__init__()

        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.max_steps = max_steps

    def forward(self, p, lambda_p):
        """Compute loss.

        Parameters
        ----------
        p : torch.Tensor
            Probability of halting of shape `(steps, batch_size)`.
        lambda_p : torch.Tensor
            Parameter determining the dynamic geometric distribution.
            of shape `(batch_size,)`.

        Returns
        -------
        loss : torch.Tensor
            Scalar representing the regularization loss.
        """
        steps, batch_size = p.shape

        # p_g = torch.zeros((self.max_steps, batch_size))
        p_g = []
        not_halted = torch.ones(batch_size).to(lambda_p.device)

        for k in range(self.max_steps):
            p_g.append(not_halted * lambda_p)
            not_halted = not_halted * (1 - lambda_p)

        p_g = torch.stack(p_g, dim=0)  # (max_steps, batch_size)

        p = p.transpose(0, 1)  # (batch_size, max_steps)
        p_g = p_g.transpose(0, 1)  # (batch_size, max_steps)

        return self.kl_div(p.log(), p_g)
