#!/bin/bash
# Job name:
#SBATCH --job-name=YOUR_JOB
#
# Partition:
#SBATCH --partition=gpu
#
# Request one node:
#SBATCH --nodes=1
#
# Specify memory for the job:
#SBATCH --mem=100G
#
# Processors per task:
#SBATCH --cpus-per-task=10
#
# GPU Ram:
#SBATCH --gres=gpu
# Wall clock limit:
#SBATCH --time=36:00:00
#
# File for output, use file or /dev/null
#SBATCH --output=/home/people/xxxx/bact-pheno/logs/%j-%x.out
#SBATCH --error=/home/people/xxxx/bact-pheno/logs/%j-%x.err#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xxxxxx@student.dtu.dk

/home/projects2/bact_pheno/venvs/ml/bin/python src/train_models.py \
    --phenotype madin_categorical_gram_stain madin_categorical_motility_binary \
    madin_carbsubs_glucose madin_carbsubs_citrate madin_carbsubs_lactose madin_carbsubs_glycerol "gideon_Beta hemolysis" madin_categorical_motility_binary "gideon_Nitrate to nitrite"  \
    madin_categorical_metabolism madin_categorical_cell_shape madin_quantitative_growth_tmp \
    --cores 10  \
    --kmer_prefixes CGTCA \
    --kmer_suffix_sizes 6 \
    --compress \
    --model_arch CNN \
    --dropout 0.2 \
    --labels_id genome_name \
    --lr 1e-3 \
    --input /home/projects2/bact_pheno/bacbench_data/ \
    --output /home/projects2/bact_pheno/bacbench_data/results/CNN/comparisons_with_bacformer/ \
    --labels_path /home/projects2/bact_pheno/bacbench_data/labels.csv 