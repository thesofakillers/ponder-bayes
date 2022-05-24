from ponderbayes.models import ponderbayes
from pyro.infer import MCMC, NUTS
import pytorch_lightning as pl


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Use MCMC to train PonderNet.")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="The seed to use for random number generation",
    )
    parser.add_argument(
        "--n-elems",
        type=int,
        default=16,
        help="Number of elements in the parity vectors",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=128,
        help="Number of hidden elements in the reccurent cell",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum number of pondering steps",
    )
    parser.add_argument(
        "--lambda-p",
        type=float,
        default=0.1,
        help="Geometric prior distribution hyperparameter",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="Regularization loss coefficient",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="whether to show the progress bar",
        default=True,
    )
    parser.add_argument(
        "--n-train-samples",
        type=int,
        default=128000,
        help="The number of training samples to comprising the dataset",
    )
    parser.add_argument(
        "--n-eval-samples",
        type=int,
        default=25600,
        help="The number of training samples to comprising the dataset",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Whether to perform 'interpolation' or 'extrapolation'",
        default="interpolation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=3,
        help="The number of workers",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        default=False,
        help="Whether to use early stopping",
    )
    parser.add_argument(
        "--n-iter", type=int, default=100000, help="Number of training steps to use"
    )
    args = parser.parse_args()

    # for reproducibility
    pl.seed_everything(args.seed)
