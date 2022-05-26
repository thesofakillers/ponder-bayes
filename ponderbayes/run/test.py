import argparse

import pytorch_lightning as pl
import re

import ponderbayes.models as models
import ponderbayes.data.datamodules as datamodules

CHECKPOINT_REGEX = r".*\/(.+?)_(.+?)_([0-9]+)\/.*"


def get_params(checkpoint_dir):
    """
    Get the params from the checkpoint directory
    """
    m = re.match(CHECKPOINT_REGEX, checkpoint_dir)
    print(m.group(1), m.group(2), int(m.group(3)))
    return m.group(1), m.group(2), int(m.group(3))


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
        model_name, mode, n_elems = get_params(args.checkpoint)
    except:
        raise ValueError(
            "Could not extract model name, data name and number of elements from checkpoint path.\
            Check that checkpoint path is of the form `<modeldir>/<model_name>_<mode_name>_<n_elems>/<version>/<checkpoint_name>.ckpt`"
        )

    # check model name
    if model_name == "pondernet":
        model_class = models.pondernet.PonderNet
    elif model_name == "groupthink":
        model_class = models.groupthink.GroupThink
    else:
        raise ValueError("Invalid `model` name in checkpoint path")

    model = model_class.load_from_checkpoint(args.checkpoint)

    # init trainer
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        enable_progress_bar=args.progress_bar,
    )

    # init data, pass None to n_train_samples because not training
    parity_datamodule = datamodules.ParityDataModule(
        n_train_samples=None,
        n_eval_samples=args.n_test_samples,
        n_elems=n_elems,
        mode=mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # test model, load directly here
    results = trainer.test(model=model, datamodule=parity_datamodule)
    print(results)
