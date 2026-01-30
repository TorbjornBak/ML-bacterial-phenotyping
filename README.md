# ML-bacterial-phenotyping
![Screenshot 2026-01-30 at 12 15 53](https://github.com/user-attachments/assets/c99e2c9b-b447-4162-b1be-4c7e11e68146)

## Getting started
### Installation
```
mamba create -n bacml python
mamba activate bacml

git clone git@github.com:TorbjornBak/ML-bacterial-phenotyping.git
cd ML-bacterial-phenotyping/
pip install -e .
```

To run an analysis, a dataset genomes in either fasta files or in a parquet format is needed.
Additionally, a metadata file in a csv format is needed.
The metadata should contain genome identifiers identical to the fasta file names or identical to identifiers in an id column in the parquet dataframe.
As an example, download a few of the parquet files and the metadata sheet from the bacformer papers phenotype prediction tasks: https://huggingface.co/datasets/macwiatrak/bacbench-phenotypic-traits-dna


### Training CNN
Trains a CNN classifier using 5-fold CV. 
```
python src/train_models_v2.py \
  --phenotype resistant_phenotype \
  --input genomes/ \
  --output results/CNN/
  --labels genomes/metadata.csv \
  --kmer_prefixes ACATG\
  --kmer_suffix_sizes 6 \
  --model_arch CNN_ONEHOT_SMALL \
  --embedding onehot \
  --id_column genome_id \
  --dna_sequence_column dna_sequence \
  --lr 3e-4 \
  --k_folds 5 \
  --group_clusters \
  --test_val_split 0.2 0.25 \
  --file_type fasta \
  --cores 4
```

### Training HistGradientBoosting model
Trains a HGBM (HistGradientBoosting) classifier, calculates SHAP values and creates feature importance plots from these.
```
python src/utilities/baselines.py  \
  --phenotype  resistant_phenotype \
  --input genomes/ \
  --output results/CNN/
  --labels genomes/metadata.csv \
  --kmer_prefix ACATG \
  --kmer_suffix_size 6 \
  --id_column genome_id \
  --dna_sequence_column dna_sequence \
  --file_type fasta \
  --embedding counts \
  --extract_feature_importance \
  --group_clusters

```
