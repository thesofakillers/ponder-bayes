"""
Credit for PonderNet, ReconstructionLoss and RegularizationLoss largely to
https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/pondernet
"""
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from torch.distributions import constraints
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.nn.module import to_pyro_module_
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam


class PonderNet(PyroModule):
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

    def __init__(self, n_elems, n_hidden=64, max_steps=20, allow_halting=False):
        super().__init__()

        self.max_steps = max_steps
        self.n_hidden = n_hidden
        self.allow_halting = allow_halting

        self.cell = PyroModule[nn.GRUCell](n_elems, n_hidden)
        self.output_layer = PyroModule[nn.Linear](n_hidden, 1)
        self.lambda_layer = PyroModule[nn.Linear](n_hidden, 1)
        self.output_layer.weight = PyroSample(dist.Normal(torch.Tensor([0.]), torch.Tensor([1.])).expand([1, n_hidden]).to_event(2))
        self.output_layer.bias = PyroSample(dist.Normal(0., 1).expand([1]).to_event(1))

    def forward(self, x, y_true=None, eval=False):
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

            # print(lambda_n)

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

        for n in range(20):
            sigma = pyro.sample(f"sigma_{n}", dist.Uniform(0., 1.))
            mean = y[n]
            with pyro.plate(f"data_{n}", x.shape[0]):
                obs = pyro.sample(f"obs_{n}", dist.Normal(mean, sigma), obs=y_true)

        if eval:
            return y
        return y, p, halting_step


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

    def forward(self, p, x_batch, y_true, model, guide):
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
            loss_per_sample = p[n] * self.loss_func(x_batch, y_true)  # (batch_size,)
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
