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
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal, AutoMultivariateNormal, init_to_mean
from pyro.optim import Adam
from ponderbayes.models import losses


class PonderBayes(PyroModule):
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
        lambda_p=0.1,
    ):
        super().__init__()

        self.n_elems = n_elems
        self.n_hidden = n_hidden
        self.max_steps = max_steps
        self.allow_halting = allow_halting
        self.beta = beta
        self.lambda_p = lambda_p

        self.cell = nn.GRUCell(n_elems, n_hidden)
        self.output_layer = PyroModule[nn.Linear](n_hidden, 2)
        self.lambda_layer = nn.Linear(n_hidden, 1)

        self.loss_reg_inst = losses.RegularizationLoss(
            lambda_p=self.lambda_p, max_steps=self.max_steps
        ).to(torch.float32)

        self.loss_rec_inst = losses.ReconstructionLoss(
            nn.BCEWithLogitsLoss(reduction="none")
        ).to(torch.float32)

        self.output_layer.weight = PyroSample(
            dist.Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
            .expand([2, n_hidden])
            .to_event(2)
        )
        self.output_layer.bias = PyroSample(
            dist.Normal(0.0, 10).expand([2]).to_event(1)
        )

    def predict(self, guide, num_samples, x_batch):
        return_sites = [f"obs_{x}" for x in range(self.max_steps)] + ["_RETURN"]
        predictive = Predictive(
            self,
            guide=guide,
            num_samples=num_samples,
            return_sites=return_sites,
        )
        predictions = predictive(x_batch)
        return predictions

    def forward(self, x, y_true=None):
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
        # moving the output layer to the right device if we haven't already
        if self.output_layer.weight.device != device:
            self.output_layer.weight = self.output_layer.weight.to(device)
            self.output_layer.bias = self.output_layer.bias.to(device)
        for n in range(1, self.max_steps + 1):
            if n == self.max_steps:
                lambda_n = x.new_ones(batch_size)  # (batch_size,)
            else:
                lambda_n = torch.sigmoid(self.lambda_layer(h))[:, 0]  # (batch_size,)

            # Store releavant outputs
            logits = self.output_layer(h)
            y_list.append(logits)  # (batch_size,)
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

            with pyro.plate(f"data_{n-1}", x.shape[0]):
                _obs = pyro.sample(
                    f"obs_{n-1}", dist.Categorical(logits=logits), obs=y_true
                )

        y = torch.stack(y_list)
        p = torch.stack(p_list)

        # for step in range(self.max_steps):
        #     sigma = pyro.sample(f"sigma_{n}", dist.Uniform(0.0, 1.0))
        #     sigma = pyro.sample(f"sigma_{n}", dist.Gamma(0.5, 1))

        # return y, p, halting_step
        # Concatinate the outputs p [max_steps,num_inputs]
        # and halting step [1,num_inputs] into the same tensor
        return torch.cat([p, halting_step.unsqueeze(0)])


class MyGuide(PyroModule):
    def __init__(self, model):
        super().__init__()
        # Let's point estimate sigma.

        self.n_hidden = model.n_hidden
        self.weights_loc = PyroParam(torch.zeros_like(model.output_layer.weight))
        self.weights_scale = PyroParam(
            torch.randn_like(model.output_layer.weight), constraint=constraints.positive
        )
        self.bias_loc = PyroParam(torch.zeros_like(model.output_layer.bias))
        self.bias_scale = PyroParam(
            torch.randn_like(model.output_layer.bias), constraint=constraints.positive
        )

    def forward(self, x, y=None):
        pyro.sample(
            "output_layer.weight",
            dist.Normal(self.weights_loc, self.weights_scale)
            .expand([2, self.n_hidden])
            .to_event(2),
        )
        pyro.sample(
            "output_layer.bias",
            dist.Normal(self.bias_loc, self.bias_scale).expand([2]).to_event(1),
        )
