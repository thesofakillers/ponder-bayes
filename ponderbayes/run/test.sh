#!/bin/bash

set -x
CHECKPOINT_FILES="./ponderbayes/run/hparams.txt"
TEST_SCRIPT="ponderbayes/run/test.py"


while read p; do
  python $TEST_SCRIPT \
    -c $p
done < "$CHECKPOINT_FILES"
