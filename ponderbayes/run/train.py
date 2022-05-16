import argparse

import pytorch_lightning as pl

import ponderbayes.models as models
import ponderbayes.data.datamodules as datamodules

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="The seed to use for random number generation",
    )
    parser.add_argument(
        "--model", type=str, default="pondernet", help="What model variant to use"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        help="path to a checkpoint from which to resume training from",
    )
    parser.add_argument(
        "--n-elems",
        type=int,
        default=64,
        help="Number of elements in the parity vectors",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=64,
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
        default=0.4,
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
        default=False,
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
        "--n-nonzero",
        type=int,
        nargs=2,
        default=(None, None),
        help="Lower and upper bound on nonzero elements in the training set",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=3,
        help="The number of workers",
    )
    args = parser.parse_args()

    # trainer config and instantiation
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        save_top_k=1, monitor="val/accuracy_halted_step", mode="max"
    )
    logger = pl.loggers.TensorBoardLogger(
        save_dir="models", name=f"{args.model}_{args.mode}"
    )
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        enable_progress_bar=args.progress_bar,
        callbacks=[ckpt_cb],
        logger=logger,
    )

    # for reproducibility
    pl.seed_everything(args.seed)

    parity_datamodule = datamodules.ParityDataModule(
        n_train_samples=args.n_train_samples,
        n_eval_samples=args.n_eval_samples,
        n_elems=args.n_elems,
        mode=args.mode,
        n_nonzero_min=args.n_nonzero[0],
        n_nonzero_max=args.n_nonzero[1],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # model instantiation
    if args.model == "pondernet":
        model_class = models.pondernet.PonderNet
    else:
        raise ValueError("Invalid `model` arg passed")
    if args.checkpoint:
        print(f"Loading from checkpoint: {args.checkpoint}")
        model = model_class.load_from_checkpoint(args.checkpoint)
    else:
        model = model_class(
            n_elems=args.n_elems,
            n_hidden=args.n_hidden,
            max_steps=args.max_steps,
            allow_halting=False,
            beta=args.beta,
            lambda_p=args.lambda_p,
        )
    # train
    trainer.fit(model, datamodule=parity_datamodule)
