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
        "--model", type=str, default="pondernet_mnist", help="What model variant to use"
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
        default=28,
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
        default=0.2,
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
        "--val-check-interval",
        type=float,
        default=0.25,
        help="Evaluate every x amount of steps, as opposed to every epoch",
    )
    parser.add_argument(
        "--n-iter", type=int, default=100000, help="Number of training steps to use"
    )
    args = parser.parse_args()

    # for reproducibility
    pl.seed_everything(args.seed)

    # model instantiation
    if args.model == "pondernet":
        model_class = models.pondernet.PonderNet
    elif args.model == "pondernet_mnist":
        model_class = models.pondernet_mnist.PonderNetMNIST
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

    # trainer config and instantiation
    cb_config = {"monitor": "val/accuracy_halted_step", "mode": "max"}
    ckpt_cb = pl.callbacks.ModelCheckpoint(save_top_k=1, **cb_config)
    callbacks = [ckpt_cb]
    if args.early_stopping:
        stop_cb = pl.callbacks.EarlyStopping(**cb_config)
        callbacks.append(stop_cb)
    logger = pl.loggers.TensorBoardLogger(
        save_dir="models", name=f"{args.model}_{args.mode}_{args.n_elems}"
    )
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        enable_progress_bar=args.progress_bar,
        max_steps=args.n_iter,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=0.5,
        val_check_interval=args.val_check_interval,
    )

    if args.mode == "interpolation":
        mnist_datamodule = datamodules.MNIST_DataModule(batch_size=args.batch_size)
    else:
        train_transform, test_transform = datamodules.get_transforms()
        mnist_datamodule = datamodules.MNIST_DataModule(
            batch_size=args.batch_size,
            train_transform=train_transform,
            test_transform=test_transform,
        )

    # train
    trainer.fit(model, datamodule=mnist_datamodule)
