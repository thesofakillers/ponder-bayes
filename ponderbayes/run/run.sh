set -x 
SEED=$RANDOM

python train.py \
	--batch-size 128 \
	--beta 0.01 \
	--eval-frequency 250 \
	--device cpu \
	--lambda-p 0.2 \
	--n-elems 4 \
	--n-iter 10000 \
	--n-hidden 128 \
	results/$SEED