#!/bin/bash
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 9
#SBATCH --mem-per-gpu 24GB
#SBATCH -p gpu
#SBATCH -t 96:00:00

module load rocky8 micromamba
micromamba activate mel_denoiser

python train.py --data_dir /gpfs/mariana/smbgroup/guitareffectsaire/data/electric/augmented_10_sec_split/train 