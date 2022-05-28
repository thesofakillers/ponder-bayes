from . import metrics


def tensor_at_halting_step(tensor, halting_step):
    """
    Indexes a step-wise batched metric by a halting step

    Parameters
    ----------
    tensor : torch.Tensor
        the metric at each step for each batch element, shape (max_steps, batch_size)
    halting_step : torch.Tensor
        the halting step for each batch element, shape (batch_size,)
    """
    tensor_halted = tensor.gather(
        dim=0,
        index=halting_step[None, :] - 1,
    )[0]

    return tensor_halted
