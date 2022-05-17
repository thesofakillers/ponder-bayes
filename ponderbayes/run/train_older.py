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
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, init_to_mean
import pyro
from pyro.infer import SVI, Trace_ELBO, Predictive
import pyro.poutine as poutine

from ponderbayes.models.ponderbayes import (
    PonderBayes
)

from ponderbayes.data.datasets import ParityDataset
from ponderbayes.models.losses import custom_loss

# @torch.no_grad()
# def evaluate(dataloader, module):
#     """Compute relevant metrics.

#     Parameters
#     ----------
#     dataloader : DataLoader
#         Dataloader that yields batches of `x` and `y`.

#     module : PonderNet
#         Our pondering network.

#     Returns
#     -------
#     metrics_single : dict
#         Scalar metrics. The keys are names and the values are `torch.Tensor`.
#         These metrics are computed as mean values over the entire dataset.

#     metrics_per_step : dict
#         Per step metrics. The keys are names and the values are `torch.Tensor`
#         of shape `(max_steps,)`. These metrics are computed as mean values over
#         the entire dataset.

#     """
#     # Imply device and dtype
#     param = next(module.parameters())
#     device, dtype = param.device, param.dtype

#     metrics_single_ = {
#         "accuracy_halted": [],
#         "halting_step": [],
#     }
#     metrics_per_step_ = {
#         "accuracy": [],
#         "p": [],
#     }

#     for x_batch, y_true_batch in dataloader:
#         x_batch = x_batch.to(device, dtype)  # (batch_size, n_elems)
#         y_true_batch = y_true_batch.to(device, dtype)  # (batch_size,)

#         y_pred_batch, p, halting_step = module(x_batch)
#         y_halted_batch = y_pred_batch.gather(dim=0, index=halting_step[None, :] - 1,)[
#             0
#         ]  # (batch_size,)

#         # Computing single metrics (mean over samples in the batch)
#         accuracy_halted = (
#             ((y_halted_batch > 0) == y_true_batch).to(torch.float32).mean()
#         )

#         metrics_single_["accuracy_halted"].append(accuracy_halted)
#         metrics_single_["halting_step"].append(halting_step.to(torch.float).mean())

#         # Computing per step metrics (mean over samples in the batch)
#         accuracy = (
#             ((y_pred_batch > 0) == y_true_batch[None, :]).to(torch.float32).mean(dim=1)
#         )

#         metrics_per_step_["accuracy"].append(accuracy)
#         metrics_per_step_["p"].append(p.mean(dim=1))

#     metrics_single = {
#         name: torch.stack(values).mean(dim=0).cpu().numpy()
#         for name, values in metrics_single_.items()
#     }

#     metrics_per_step = {
#         name: torch.stack(values).mean(dim=0).cpu().numpy()
#         for name, values in metrics_per_step_.items()
#     }

#     return metrics_single, metrics_per_step

@torch.no_grad()
def evaluate(dataloader, module,guide=None):
    """Compute relevant metrics.

    Parameters
    ----------
    dataloader : DataLoader
        Dataloader that yields batches of `x` and `y`.

    module : PonderNet
        Our pondering network.

    Returns
    -------
    metrics_single : dict
        Scalar metrics. The keys are names and the values are `torch.Tensor`.
        These metrics are computed as mean values over the entire dataset.

    metrics_per_step : dict
        Per step metrics. The keys are names and the values are `torch.Tensor`
        of shape `(max_steps,)`. These metrics are computed as mean values over
        the entire dataset.

    """
    # Imply device and dtype
    param = next(module.parameters())
    device, dtype = param.device, param.dtype

    metrics_single_ = {
        "accuracy_halted": [],
        "accuracy_halted_predictive": [],
        "halting_step": [],
    }
    metrics_per_step_ = {
        "accuracy": [],
        "p": [],
    }

    for x_batch, y_true_batch in dataloader:
        x_batch = x_batch.to(device, dtype)  # (batch_size, n_elems)
        y_true_batch = y_true_batch.to(device, dtype)  # (batch_size,)

        # predictive = Predictive(module, guide=guide, num_samples=3,return_sites=["_RETURN"])
        predictive = Predictive(module, guide=guide, num_samples=3)
        predictions = predictive(x_batch)
        y_pred_batch, p, halting_step = module(x_batch)
        y_halted_batch = y_pred_batch.gather(dim=0, index=halting_step[None, :] - 1,)[
            0
        ]  # (batch_size,)

        y_halted_predictive = torch.zeros_like(y_halted_batch)
        for sample in range(x_batch.shape[0]):
            h_step = int(halting_step[sample])
            y_halted_predictive[sample] = predictions[f'obs_{h_step}'][:,sample].mean()
        
        # Computing single metrics (mean over samples in the batch)
        accuracy_halted = (
            ((y_halted_batch > 0) == y_true_batch).to(torch.float32).mean()
        )
        
        accuracy_halted_predictive = (
            ((y_halted_predictive > 0) == y_true_batch).to(torch.float32).mean()
        )
        
        metrics_single_["accuracy_halted"].append(accuracy_halted)
        metrics_single_["accuracy_halted_predictive"].append(accuracy_halted_predictive)
        metrics_single_["halting_step"].append(halting_step.to(torch.float).mean())

        # Computing per step metrics (mean over samples in the batch)
        accuracy = (
            ((y_pred_batch > 0) == y_true_batch[None, :]).to(torch.float32).mean(dim=1)
        )

        metrics_per_step_["accuracy"].append(accuracy)
        metrics_per_step_["p"].append(p.mean(dim=1))

    metrics_single = {
        name: torch.stack(values).mean(dim=0).cpu().numpy()
        for name, values in metrics_single_.items()
    }

    metrics_per_step = {
        name: torch.stack(values).mean(dim=0).cpu().numpy()
        for name, values in metrics_per_step_.items()
    }

    return metrics_single, metrics_per_step

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
        "--log_folder",
        type=str,
        default="results/666",
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
        default=10_00,
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
    # log_folder = 'results/666'
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
    model = PonderBayes(
        n_elems=args.n_elems,
        n_hidden=args.n_hidden,
        max_steps=args.max_steps,
    )
    guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)
    # module = module.to(device, dtype)

    # loss_reg_inst = RegularizationLoss(
    #     lambda_p=args.lambda_p,
    #     max_steps=args.max_steps,
    # ).to(device, dtype)

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
    
    for epoch in range(12):
        iterator = tqdm(enumerate(dataloader_train), total=args.n_iter)
        for step, (x_batch, y_true_batch) in iterator:
            x_batch = x_batch.to(device, dtype)
            y_true_batch = y_true_batch.to(device, dtype)

            loss = svi.step(x_batch, y_true_batch)
           
            writer.add_scalar(
                            "training/epoch{epoch}/loss",
                            loss, # print(loss)
                            step
                        )
            # # Evaluation
            if step % args.eval_frequency == 0:
                
                model.eval()

                for dataloader_name, dataloader in eval_dataloaders.items():
                    metrics_single, metrics_per_step = evaluate(dataloader,model,guide)
                    # fig_dist = plot_distributions(
                    #     loss_reg_inst.p_g.cpu().numpy(),
                    #     metrics_per_step["p"],
                    # )
                    # writer.add_figure(f"distributions/{dataloader_name}", fig_dist, step)

                    fig_acc = plot_accuracy(metrics_per_step["accuracy"])
                    writer.add_figure(f"accuracy_per_step/{dataloader_name}", fig_acc, step)

                    for metric_name, metric_value in metrics_single.items():
                        writer.add_scalar(
                            f"{metric_name}/epoch{epoch}/{dataloader_name}",
                            metric_value,
                            step,
                        )

                torch.save(model, log_folder / "checkpoint.pth")

                model.train()


if __name__ == "__main__":
    main()
