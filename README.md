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
  --config results/reviewer_benchmark_jaccard_096/resolved_run_manifest.json
```

Split preparation reads genomes incrementally and caches completed Sourmash
sketches in a `sketches/` directory next to the split manifest. A rerun with
the same genome files and sketch settings reuses the sketches in that directory.
The primary configuration joins every genome pair with Jaccard similarity
at least 0.96 using connected components. It checks every pair and keeps
memory proportional to the number of genomes beyond the sketches.

For the 0.90 sensitivity run, copy the analysis config, set
`clustering.min_similarity` to `0.90`, and use a separate `output_dir`, such
as `results/reviewer_benchmark_jaccard_090`. Prepare its own condition and
split manifests. Check that each phenotype has five folds containing all
classes before running the full matrix, and use the same split manifest for
every full and filtered condition within each run.

Use the resolved manifest for execution because it records the code version,
absolute input/output paths, sampled-prefix manifest, and manifest hashes.
Each condition is independently runnable (and suitable for a SLURM array):

```bash
python src/reviewer_benchmark.py run-condition \
  --config results/reviewer_benchmark_jaccard_096/resolved_run_manifest.json \
  --condition-manifest results/reviewer_benchmark_jaccard_096/conditions.pilot.csv \
  --condition-index 0 \
  --split-manifest results/reviewer_benchmark_jaccard_096/split_manifest.csv
```

Add `--workers 2` to count k-mers in two CPU processes. The default is one
worker. Run one condition at a time on memory-limited machines and use the
same worker count for comparisons of wall time and RAM. The resource table
records the number of counting workers used.

Use `run-conditions` for a progress bar across the full manifest or a subset.
The terminal also shows progress through genome counting and fold evaluation
for the active condition. For the four-condition gentamicin pilot:

```bash
run=results/reviewer_benchmark_jaccard_096
.venv/bin/python src/reviewer_benchmark.py run-conditions \
  --config "$run/resolved_run_manifest.json" \
  --condition-manifest "$run/conditions.csv" \
  --split-manifest "$run/split_manifest.csv" \
  --condition-indices 0 73 63 64 --workers 2
```

Omit `--condition-indices` to run every row. `--no-progress` hides the bars
and keeps the start and completion messages.

To fit the project's deterministic HistGradientBoosting model on an existing
prefix-filtered matrix without repeating tokenization, add
`--reuse-features --model hgbm`. HGBM outputs are written below the condition's
`models/hist_gradient_boosting` directory and do not replace logistic-regression
results. Full-representation HGBM runs are intentionally rejected because the
dense matrix would be impractical.

Submit from the cluster checkout root through `shell_scripts/reviewer_benchmark.sh`.
Set `REVIEWER_REPO_ROOT` to the absolute checkout path if submitting from elsewhere.
Prepare the resolved manifest on the cluster so its absolute paths point to
cluster storage. The script's optional fourth and fifth arguments select the model and reuse
mode, for example `hgbm true`; select only prefix-condition array indices for
that model. The notebook `notebooks/reviewer_benchmark.ipynb` reads only the
saved outputs; it does not retrain models.

### Transformer pilot

`src/reviewer_transformer_pilot.py` trains a Transformer on the ordered
suffix tokens of one prefix-filtered condition, using the same frozen folds,
metrics, and output layout as the RNN pilot (`models/transformer/<run>`).
Attention uses PyTorch's fused scaled-dot-product kernels, so memory grows
linearly with sequence length; it trains on a 16 GB laptop CPU without
thinning a typical ACATG/k=11 genome (~3,200 tokens):

```bash
PYTHONPATH=src .venv/bin/python src/reviewer_transformer_pilot.py \
  --config results/reviewer_benchmark/resolved_run_manifest.json \
  --split-manifest results/reviewer_benchmark/split_manifest.csv \
  --prefix ACATG --total-k 11 --folds 0 1 2 3 4
```

`--max-tokens` caps each genome by even thinning (default 4096);
`--num-layers 0` gives an attention-free embedding-bag ablation. Use CPU or
CUDA: Apple MPS materialises the full attention matrix and runs out of memory
at genome lengths.

The general cross-validation trainer also supports the bounded configuration.
It defaults to 2,048 uniformly sampled tokens and batch size 2, and currently
requires integer embeddings:

```bash
.venv/bin/python src/train_models_v2.py \
  --phenotype resistant_phenotype \
  --input genomes/ --output results/transformer/ \
  --labels genomes/metadata.csv \
  --id_column genome_id --file_type fasta \
  --kmer_prefixes ACATG --kmer_suffix_sizes 6 \
  --model_arch TRANSFORMER --embedding integer \
  --learning_rates 3e-4 --device auto --cores 2 \
  --transformer_max_tokens 2048 --transformer_batch_size 2
```

Add `--reembed` on the first run. Lower the token cap to 1024 or the batch size
to 1 if other processes leave little free memory.

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
