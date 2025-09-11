#!/bin/bash
# Job name:
#SBATCH --job-name=download_genomes
#
# Partition:
#SBATCH --partition=cpu
#
# Request one node:
#SBATCH --nodes=1
#
# Specify memory for the job (example):
#SBATCH --mem=10G
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=00:02:00
#
# File for output, use file or /dev/null
#SBATCH -o /home/people/s203555/bact-pheno
#
## Command(s) to run (example):

/home/ctools/opt/anaconda3_2023-3-1/bin/python3 download_genomes.py 2000
