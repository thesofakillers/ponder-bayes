set -x 
SEED=$RANDOM

python3 train.py \
		--batch-size 128 \
		--beta 0.01 \
		--device cuda \
		--eval-frequency 500 \
		--n-iter 10000 \
		--n-hidden 128 \
		--lambda-p 0.4 \
		--n-elems 4 \
        --seed $SEED \
		results/experiment_a/$SEED