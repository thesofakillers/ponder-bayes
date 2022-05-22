from math import inf
import torch
import torch.nn as nn
import pyro.poutine as poutine

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
        p = p + torch.finfo(torch.float32).eps
        p_g_batch = self.p_g[None, :steps].expand_as(p)  # (batch_size, max_steps)

        return self.kl_div(p.log(), p_g_batch)



def custom_loss(model, guide, *args, **kwargs):
    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
    model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace(
        *args, **kwargs
    )

    elbo = 0.0
    # P is only the first max_steps rows returned values
    p = model_trace.nodes["_RETURN"]["value"][:model.max_steps]

    # print(model_trace.nodes)

    # We are interested in obs_step and output layer weights and biases
    for site in model_trace.nodes.values():
        if site["type"] == "sample":
            if "obs" in site["name"]:
                step = int(site["name"].split("_")[-1])
                score = site["fn"].log_prob(site["value"]) * p[step]
            elif "halt" not in site["name"]:
                score = site["fn"].log_prob(site["value"])
            elbo += score.mean()

    for site in guide_trace.nodes.values():
        if site["type"] == "sample":
            if "obs" in site["name"]:
                step = int(site["name"].split("_")[-1])
                score = site["fn"].log_prob(site["value"]) * p[step]
            else:
                score = site["fn"].log_prob(site["value"])
            elbo -= score.mean()

    reg_loss = model.loss_reg_inst(p) * model.beta
    # reg_loss = 0.0
    return -elbo + reg_loss
