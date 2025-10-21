#!/bin/bash
# Job name:
#SBATCH --job-name=train_RNN
#
# Partition:
#SBATCH --partition=gpu
#
# Request one node:
#SBATCH --nodes=1
#
# Specify memory for the job (example):
#SBATCH --mem=80G
#
# Processors per task:
#SBATCH --cpus-per-task=20
#
# GPU Ram:
#SBATCH --gres=gpu
# Wall clock limit:
#SBATCH --time=10:00:00
#
# File for output, use file or /dev/null
#SBATCH --output=/home/people/s203555/bact-pheno/logs/%x-%j.out
#SBATCH --error=/home/people/s203555/bact-pheno/logs/%x-%j.err#

/home/projects2/bact_pheno/venvs/ml/bin/python scripts/CNN_v2.py --REEMBED=FALSE --DROPOUT=0.2  --PHENOTYPE=madin_categorical_gram_stain   --CORES=20 --MODEL_ARCH=RNN

