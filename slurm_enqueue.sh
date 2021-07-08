#!/bin/bash

if [ $# -lt 2 ]; then
	echo "USAGE: slurm_enqueue model version"
    echo "       Example: $0 fasterRCNN v1.1"
    exit
fi

#source ~/venv/ai4agr/bin/activate

#srun --gres=gpu:1 --mem=16G --time=24:00:00 python3 -u ~/grapes_detection/grapes_torch/torch_nets/train_cascade.py --version $2 --model $1 > ~/grapes_detection/logs/$1_$2.log 2>&1
if [ "$1" = "fasterRCNN" ]; then
    echo "faster rcnn"    
    srun --gres=gpu:1 --mem=32G --time=24:00:00 ./train_faster.sh $2 > ../logs/fasterRCNN_$2.log 2>&1
elif [ "$1" = "ResidualUNet" ]; then
    echo "residual unet"
    srun --gres=gpu:1 --mem=32G --time=24:00:00 --exclude=gpic09 ./train_unet.sh $2 > ../logs/ResidualUNet_$2.log 2>&1
else
    echo "Wrong model name"
    exit 1
fi
