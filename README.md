# ML-bacterial-phenotyping
![Screenshot 2026-01-30 at 12 15 53](https://github.com/user-attachments/assets/c99e2c9b-b447-4162-b1be-4c7e11e68146)

## Getting started

### Local Installation
```
mamba create -n bacml python
mamba activate bacml

git clone git@github.com:TorbjornBak/ML-bacterial-phenotyping.git
cd ML-bacterial-phenotyping/
pip install -e .
```

To run an analysis, a set of genomes in either fasta files or in a parquet format is needed.
Additionally, a metadata file in a csv format is needed.
The metadata should contain genome identifiers identical to the fasta file names or identical to identifiers in an id column in the parquet dataframe.

As an example, download a few of the parquet files and the metadata sheet from the bacformer papers phenotype prediction tasks: https://huggingface.co/datasets/macwiatrak/bacbench-phenotypic-traits-dna

### Reviewer benchmark

The sparse full-versus-prefix benchmark is driven by
`configs/reviewer_benchmark.example.json`. Copy and edit that file, then
freeze the pilot conditions and the full-genome cluster/fold assignments:

```bash
python src/reviewer_benchmark.py prepare --config configs/reviewer_benchmark.json --pilot
python src/reviewer_benchmark.py prepare-splits \
  --config results/reviewer_benchmark/resolved_run_manifest.json
```

Use the resolved manifest for execution because it records the code version,
absolute input/output paths, sampled-prefix manifest, and manifest hashes.
Each condition is independently runnable (and suitable for a SLURM array):

```bash
python src/reviewer_benchmark.py run-condition \
  --config results/reviewer_benchmark/resolved_run_manifest.json \
  --condition-manifest results/reviewer_benchmark/conditions.pilot.csv \
  --condition-index 0 \
  --split-manifest results/reviewer_benchmark/split_manifest.csv
```

To fit the project's deterministic HistGradientBoosting model on an existing
prefix-filtered matrix without repeating tokenization, add
`--reuse-features --model hgbm`. HGBM outputs are written below the condition's
`models/hist_gradient_boosting` directory and do not replace logistic-regression
results. Full-representation HGBM runs are intentionally rejected because the
dense matrix would be impractical.

Submit the same inputs through `shell_scripts/reviewer_benchmark.sh` for a
cluster run. Its optional fourth and fifth arguments select the model and reuse
mode, for example `hgbm true`; select only prefix-condition array indices for
that model. The notebook `notebooks/reviewer_benchmark.ipynb` reads only the
saved outputs; it does not retrain models.

### Installation on Healthtech Cluster

See this guide to install on DTU's HealthTech Cluster
[HealthTech Docs](https://github.com/TorbjornBak/ML-bacterial-phenotyping/blob/main/docs/healthtech.md)


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
  --embedding frequency \
  --submodule feature_importance \
  --group_clusters

```
