set -x 
SEED=$RANDOM

python train_older.py \
	--batch-size 128 \
	--beta 0.01 \
	--eval-frequency 500 \
	--device cpu \
	--lambda-p 0.2 \
	--n-elems 4 \
	--n-iter 10000 \
	--n-hidden 128 \
	--log_folder results/$SEED