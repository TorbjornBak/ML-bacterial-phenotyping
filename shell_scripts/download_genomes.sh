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
#SBATCH --time=02:00:00
#
# File for output, use file or /dev/null
#SBATCH --output=/home/people/s203555/bact-pheno/logs/%x-%j.out
#SBATCH --error=/home/people/s203555/bact-pheno/logs/%x-%j.err#

mkdir -p /home/people/s203555/bact-pheno/logs

/home/ctools/opt/anaconda3_2023-3-1/bin/python3 scripts/download_genomes.py 2000 /home/people/s203555/bact-pheno/ML-bacterial-phenotyping/downloads/genome_summary /home/projects2/s203555/bv-brc-data
