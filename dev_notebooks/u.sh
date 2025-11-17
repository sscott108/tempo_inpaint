#!/bin/bash
#SBATCH -o u.out
#SBATCH -e u.err
#SBATCH -p carlsonlab-gpu
#SBATCH -c 4 # CPUs, adjust as needed
#SBATCH --mem-per-cpu=20G # adjust as needed
#SBATCH --gres=gpu:1

python PConv_UNET_cookbook.py
