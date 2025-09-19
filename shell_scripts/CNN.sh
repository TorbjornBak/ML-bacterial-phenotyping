#!/bin/bash
# Job name:
#SBATCH --job-name=download_genomes
#
# Partition:
#SBATCH --partition=gpu
#
# Request one node:
#SBATCH --nodes=1
#
# Specify memory for the job (example):
#SBATCH --mem=40G
#
# Processors per task:
#SBATCH --cpus-per-task=10
#
# GPU Ram:
#SBATCH --gres=shard:10
# Wall clock limit:
#SBATCH --time=00:30:00
#
# File for output, use file or /dev/null
#SBATCH --output=/home/people/s203555/bact-pheno/logs/%x-%j.out
#SBATCH --error=/home/people/s203555/bact-pheno/logs/%x-%j.err#


~/venvs/ml/bin/python scripts/CNN_v2.py --REEMBED=TRUE --DROPOUT=0.5 --EPOCHS=20 --LR=0.001 --BATCH_SIZE=50 --PHENOTYPE=madin_categorical_gram_stain  --EPOCHS=40 --K_SIZE=4 --KMER_PREFIX=CGTC --CORES=10 

