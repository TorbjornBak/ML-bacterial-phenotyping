# ML-bacterial-phenotyping

## Getting started
### Installation
```
mamba create -n bacml python
mamba activate bacml

git clone git@github.com:TorbjornBak/ML-bacterial-phenotyping.git
cd ML-bacterial-phenotyping/
pip install -e .
```

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

### Training baseline
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
