#!/bin/bash

module purge
module load 2021
module load Anaconda3/2021.05

source activate ponderbayes
tensorboard --logdir models/
