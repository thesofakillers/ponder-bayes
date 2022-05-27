#!/bin/bash

# SEEDS_FILE=./lisa/seeds.txt
for s in $(seq 1 5)
do
echo $s
# comment out the final line if you only care about seed 42
python ponderbayes/run/train_mnist.py --model pondernet_mnist \
  --mode interpolation \
  --n-elems 28 \
  --n-hidden 128 \
  --lambda-p 0.1 \
  --beta 0.01 --batch-size 128 \
  --num-workers 3 \
  --n-iter 20000 \
  --n-train-samples 1000000 \
  --n-eval-samples 200000 \
  --val-check-interval 0.25 \
  --mode interpolation \
  --seed $s
python ponderbayes/run/train_mnist.py --model pondernet_mnist \
  --mode interpolation \
  --n-elems 28 \
  --n-hidden 128 \
  --lambda-p 0.1 \
  --beta 0.01 --batch-size 128 \
  --num-workers 3 \
  --n-iter 20000 \
  --n-train-samples 1000000 \
  --n-eval-samples 200000 \
  --val-check-interval 0.25 \
  --mode extrapolation \
  --seed $s
done