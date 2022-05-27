#!/bin/bash

set -x
CHECKPOINT_FILE="./ponderbayes/run/hparams.txt"
TEST_SCRIPT="./ponderbayes/run/test.py"


if [ $# -ne 1 ]; then
    echo "Please specify the model to collect the checkpoints for!"
    echo "Example: bash test.sh pondernet"
    exit 1
fi

CURRENT_DIRECTORY=$(basename "$(pwd)")
# Script should be called in the base directory
if [[ "$CURRENT_DIRECTORY" == "run" ]]; then
    cd ../..
elif [[ "$CURRENT_DIRECTORY" == "ponderbayes" ]]; then
    cd ..
fi


# finds all checkpoint files for a particular model
find ./models -name "*.ckpt" | grep $1 > $CHECKPOINT_FILE


# tests all models in the checkpoint file
while read p; do
  python $TEST_SCRIPT \
    -c $p
done < "$CHECKPOINT_FILE"
