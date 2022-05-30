import argparse

import pytorch_lightning as pl
import re

import ponderbayes.models as models
import ponderbayes.data.datamodules as datamodules

CHECKPOINT_REGEX = r".*\/(.+)_(.+?)_([0-9]+)\/.*?_([0-9]+).+"


def get_params(checkpoint_dir):
    """
    Get the params from the checkpoint directory
    """
    m = re.match(CHECKPOINT_REGEX, checkpoint_dir)
    return {
        "name": m.group(1),
        "mode": m.group(2),
        "n_elems": int(m.group(3)),
        "version": int(m.group(4)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a pondernet checkpoint")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="The seed to use for random number generation",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="path (relative to root) to a checkpoint to evaluate",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="whether to show the progress bar",
        default=True,
    )
    parser.add_argument(
        "--n-test-samples",
        type=int,
        default=25600,
        help="The number of testing samples to comprise the dataset",
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
    args = parser.parse_args()

    # for reproducibility
    pl.seed_everything(args.seed)

    # extract data params from model path
    print(f"Loading from checkpoint: {args.checkpoint}")
    try:
        hparams = get_params(args.checkpoint)
    except:
        raise ValueError(
            "Could not extract model name, data name and number of elements from checkpoint path.\
            Check that checkpoint path is of the form `<modeldir>/<model_name>_<mode_name>_<n_elems>/<version>/<checkpoint_name>.ckpt`"
        )

    # check model name
    if hparams["name"] == "pondernet":
        model_class = models.pondernet.PonderNet
    elif hparams["name"] == "groupthink":
        model_class = models.groupthink.GroupThink
    elif hparams["name"] == "pondernet_mnist":
        model_class = models.pondernet_mnist.PonderNetMNIST
    elif hparams["name"] == "aRGT":
        model_class = models.aRGT.RationalGroupThink
    elif hparams["name"] == "RGT":
        model_class = models.RGT.RationalGroupThink
    elif hparams["name"] == "lambdaGT":
        model_class = models.lagt.LambdaGroupThink
    else:
        raise ValueError("Invalid `model` name in checkpoint path")

    model = model_class.load_from_checkpoint(args.checkpoint)

    # init trainer and logger
    logger = pl.loggers.TensorBoardLogger(
        save_dir="models",
        name=f'{hparams["name"]}_{hparams["mode"]}_{hparams["n_elems"]}',
        version=hparams["version"],
    )
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        enable_progress_bar=args.progress_bar,
        logger=logger,
    )

    # init data
    if hparams["name"] == "pondernet_mnist":
        if hparams["mode"] == "interpolation":
            datamodule = datamodules.MNIST_DataModule(batch_size=args.batch_size)
        else:
            train_transform, test_transform = datamodules.get_transforms()
            datamodule = datamodules.MNIST_DataModule(
                batch_size=args.batch_size,
                train_transform=train_transform,
                test_transform=test_transform,
            )
    else:
        # pass None to n_train_samples because not training
        datamodule = datamodules.ParityDataModule(
            n_train_samples=None,
            n_eval_samples=args.n_test_samples,
            n_elems=hparams["n_elems"],
            mode=hparams["mode"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # test model, load directly here
    results = trainer.test(model=model, datamodule=datamodule)
