from argparse import ArgumentParser
import json
import pathlib
from pickletools import optimize

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pyro.infer.autoguide import AutoDiagonalNormal
import pyro
from pyro.infer import SVI, Trace_ELBO
import pyro.poutine as poutine

from ponderbayes.models.ponderbayes import (
    PonderNet,
    ReconstructionLoss,
    RegularizationLoss,
)

from ponderbayes.data.datasets import ParityDataset
from pyro.infer import Predictive



@torch.no_grad()
def evaluate(model, guide, dataloader):
    def summary(samples):
        site_stats = {}
        for k, v in samples.items():
            site_stats[k] = {
                "mean": torch.mean(v, 0),
            }
        return site_stats

    predictive = Predictive(model, guide=guide, num_samples=1,
                            return_sites=("linear.weight", "obs", "_RETURN"))

    
    for x_batch, y_true_batch in dataloader:
        samples = predictive(x_batch, eval=True)
        pred_summary = summary(samples)
        print(pred_summary)


def custom_loss(model, guide, *args, **kwargs):
    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
    model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)

    elbo = 0.0
    y, p, halting_step = model_trace.nodes['_RETURN']['value']

    count = 0
    for site in model_trace.nodes.values():
        if site["type"] == "sample" and 'obs' in site["name"]:
            score = site['fn'].log_prob(site['value']) * p[count]
            elbo += score.mean()
            count += 1

    count = 0
    for site in guide_trace.nodes.values():
        if site["type"] == "sample" and 'obs' in site["name"]:
            score = site['fn'].log_prob(site['value']) * p[count]
            elbo -= score.mean()
            count += 1
    
    return -elbo 



def plot_distributions(target, predicted):
    """Create a barplot.

    Parameters
    ----------
    target, predicted : np.ndarray
        Arrays of shape `(max_steps,)` representing the target and predicted
        probability distributions.

    Returns
    -------
    matplotlib.Figure
    """
    support = list(range(1, len(target) + 1))

    fig, ax = plt.subplots(dpi=140)

    ax.bar(
        support,
        target,
        color="red",
        label=f"Target - Geometric({target[0].item():.2f})",
    )

    ax.bar(
        support,
        predicted,
        color="green",
        width=0.4,
        label="Predicted",
    )

    ax.set_ylim(0, 0.6)
    ax.set_xticks(support)
    ax.legend()
    ax.grid()

    return fig


def plot_accuracy(accuracy):
    """Create a barplot representing accuracy over different halting steps.

    Parameters
    ----------
    accuracy : np.array
        1D array representing accuracy if we were to take the output after
        the corresponding step.

    Returns
    -------
    matplotlib.Figure
    """
    support = list(range(1, len(accuracy) + 1))

    fig, ax = plt.subplots(dpi=140)

    ax.bar(
        support,
        accuracy,
        label="Accuracy over different steps",
    )

    ax.set_ylim(0, 1)
    ax.set_xticks(support)
    ax.legend()
    ax.grid()

    return fig


def main(argv=None):
    """CLI for training."""
    parser = ArgumentParser()

    parser.add_argument(
        "log_folder",
        type=str,
        help="Folder where tensorboard logging is saved",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="Regularization loss coefficient",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices={"cpu", "cuda"},
        default="cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=10_000,
        help="Evaluation is run every `eval_frequency` steps",
    )
    parser.add_argument(
        "--lambda-p",
        type=float,
        default=0.4,
        help="True probability of success for a geometric distribution",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=1_000_000,
        help="Number of gradient steps",
    )
    parser.add_argument(
        "--n-elems",
        type=int,
        default=64,
        help="Number of elements",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed of run",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=64,
        help="Number of hidden elements in the reccurent cell",
    )
    parser.add_argument(
        "--n-nonzero",
        type=int,
        nargs=2,
        default=(None, None),
        help="Lower and upper bound on nonzero elements in the training set",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum number of pondering steps",
    )

    # Parameters
    args = parser.parse_args(argv)
    print(args)

    device = torch.device(args.device)
    dtype = torch.float32
    n_eval_samples = 1000
    batch_size_eval = 50

    if args.n_nonzero[0] is None and args.n_nonzero[1] is None:
        threshold = int(0.3 * args.n_elems)
        range_nonzero_easy = (1, threshold)
        range_nonzero_hard = (args.n_elems - threshold, args.n_elems)
    else:
        range_nonzero_easy = (1, args.n_nonzero[1])
        range_nonzero_hard = (args.n_nonzero[1] + 1, args.n_elems)

    # Tensorboard
    log_folder = pathlib.Path(args.log_folder)
    writer = SummaryWriter(log_folder)
    writer.add_text("parameters", json.dumps(vars(args)))

    # Prepare data
    dataloader_train = DataLoader(
        ParityDataset(
            n_samples=args.batch_size * args.n_iter,
            n_elems=args.n_elems,
            n_nonzero_min=args.n_nonzero[0],
            n_nonzero_max=args.n_nonzero[1],
        ),
        batch_size=args.batch_size,
    )  # consider specifying `num_workers` for speedups
    eval_dataloaders = {
        "test": DataLoader(
            ParityDataset(
                n_samples=n_eval_samples,
                n_elems=args.n_elems,
                n_nonzero_min=args.n_nonzero[0],
                n_nonzero_max=args.n_nonzero[1],
            ),
            batch_size=batch_size_eval,
        ),
        f"{range_nonzero_easy[0]}_{range_nonzero_easy[1]}": DataLoader(
            ParityDataset(
                n_samples=n_eval_samples,
                n_elems=args.n_elems,
                n_nonzero_min=range_nonzero_easy[0],
                n_nonzero_max=range_nonzero_easy[1],
            ),
            batch_size=batch_size_eval,
        ),
        f"{range_nonzero_hard[0]}_{range_nonzero_hard[1]}": DataLoader(
            ParityDataset(
                n_samples=n_eval_samples,
                n_elems=args.n_elems,
                n_nonzero_min=range_nonzero_hard[0],
                n_nonzero_max=range_nonzero_hard[1],
            ),
            batch_size=batch_size_eval,
        ),
    }

    # Model preparation
    model = PonderNet(
        n_elems=args.n_elems,
        n_hidden=args.n_hidden,
        max_steps=args.max_steps,
    )
    guide = AutoDiagonalNormal(model)
    # module = module.to(device, dtype)

    loss_reg_inst = RegularizationLoss(
        lambda_p=args.lambda_p,
        max_steps=args.max_steps,
    ).to(device, dtype)

    # Optimizer
    adam = pyro.optim.Adam({"lr": 0.03})
    # loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

    # Loss preparation
    # loss_rec_inst = ReconstructionLoss(loss_fn).to(
    #     device, dtype
    # )
    svi = SVI(model, guide, adam, loss=custom_loss)

    # Training and evaluation loops
    pyro.clear_param_store()
    iterator = tqdm(enumerate(dataloader_train), total=args.n_iter)
    for step, (x_batch, y_true_batch) in iterator:
        x_batch = x_batch.to(device, dtype)
        y_true_batch = y_true_batch.to(device, dtype)

        loss = svi.step(x_batch, y_true_batch)
        print(loss)

        # # Evaluation
        if step % args.eval_frequency == 0:
            evaluate(model, guide, eval_dataloaders['test'])


if __name__ == "__main__":
    main()
