

/home/projects2/bact_pheno/venvs/ml/bin/python scripts/train_models.py --PHENOTYPE=madin_categorical_gram_stain --CORES=10 --KMER_PREFIXES=CGTCAC,CGTCA --KMER_SUFFIX_SIZES=1,2,4,6 --COMPRESS=TRUE --MODEL_ARCH=RNN --DROPOUT=0.2 --DATA_OUTPUT=/home/projects2/bact_pheno/bacbench_data/results/RNN/compressed_full_run

/home/projects2/bact_pheno/venvs/ml/bin/python scripts/train_models.py --PHENOTYPE=madin_categorical_gram_stain --CORES=10 --KMER_PREFIXES=CGTCAC,CGTCA --KMER_SUFFIX_SIZES=8,10,12 --COMPRESS=TRUE --MODEL_ARCH=RNN --DROPOUT=0.2 --DATA_OUTPUT=/home/projects2/bact_pheno/bacbench_data/results/RNN/compressed_full_run

/home/projects2/bact_pheno/venvs/ml/bin/python scripts/train_models.py --PHENOTYPE=madin_categorical_gram_stain --CORES=10 --KMER_PREFIXES=CGTCACA,CGTCAC --KMER_SUFFIX_SIZES=8,10,12 --COMPRESS=TRUE --MODEL_ARCH=RNN --DROPOUT=0.2 --DATA_OUTPUT=/home/projects2/bact_pheno/bacbench_data/results/RNN/compressed_full_run

/home/projects2/bact_pheno/venvs/ml/bin/python scripts/train_models.py --PHENOTYPE=madin_categorical_gram_stain --CORES=10 --KMER_PREFIXES=CGTCA--KMER_SUFFIX_SIZES=8,10,12 --COMPRESS=TRUE --MODEL_ARCH=RNN --DROPOUT=0.2 --DATA_OUTPUT=/home/projects2/bact_pheno/bacbench_data/results/RNN/compressed_full_run


/home/projects2/bact_pheno/venvs/ml/bin/python scripts/train_models.py --PHENOTYPE=madin_categorical_gram_stain --CORES=10 --KMER_PREFIXES=CGTCA --KMER_SUFFIX_SIZES=1,2,4,6 --COMPRESS=FALSE --MODEL_ARCH=RNN --DROPOUT=0.2 --DATA_OUTPUT=/home/projects2/bact_pheno/bacbench_data/results/RNN/uncompressed_full_run

/home/projects2/bact_pheno/venvs/ml/bin/python scripts/train_models.py --PHENOTYPE=madin_categorical_gram_stain --CORES=10 --KMER_PREFIXES=CGTCACA,CGTCAC --KMER_SUFFIX_SIZES=8,10,12 --COMPRESS=FALSE --MODEL_ARCH=RNN --DROPOUT=0.2 --DATA_OUTPUT=/home/projects2/bact_pheno/bacbench_data/results/RNN/uncompressed_full_run

/home/projects2/bact_pheno/venvs/ml/bin/python scripts/train_models.py --PHENOTYPE=madin_categorical_gram_stain --CORES=10 --KMER_PREFIXES=CGTCA --KMER_SUFFIX_SIZES=8,10,12 --COMPRESS=FALSE --MODEL_ARCH=RNN --DROPOUT=0.2 --DATA_OUTPUT=/home/projects2/bact_pheno/bacbench_data/results/RNN/uncompressed_full_run