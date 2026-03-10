# Getting started with the repository on DTU's Health Tech Cluster
To run the code in the repository on a cluster like DTU's Health Tech cluster, please follow this guide step by step (or skip if you know what you're doing :))

# Logging in to the Health Tech cluster
To log in, type the following in the terminal, replace with your student number:
```bash
ssh s123456@login.healthtech.dtu.dk
```

Then type “yes”, then type your DTU password, then authenticate with your Authenticator app on your phone. 
Now you should be logged in.

Here is DTU health tech’s guide to the cluster for reference (you don't need to look at it now):
https://nexus.healthtech.dtu.dk/wiki/books/onboarding-information/page/beginners-guide-to-the-health-tech-cluster

# Cloning the code repository
Make sure you are in your home directory (just type `cd` to go to your home directory if you're in doubt)
```bash
git clone https://github.com/TorbjornBak/ML-bacterial-phenotyping.git
cd ML-bacterial-phenotyping 
```

## Make a directory for the project in projects2
```bash
mkdir /home/projects2/your_project_name/ # replace with some other name
mkdir /home/projects2/your_project_name/venvs # Make this directory to store your virtual environments
```


# Creating a virtual environment on the cluster
To create a virtual environment for a project, you need to choose a base Python installation. I recommend using this general one for cpu based workloads, which should look something like: `/home/ctools/opt/anaconda3_2023-3-1/`. 
(And you will need this one instead for the gpu nodes: `/home/local/tools/anaconda3-24.10.1/bin/python3`)


Create the virtual environment using system site packages in the specificed directory, on an interactive node (like node07):
```bash
ssh node07
/home/ctools/opt/anaconda3_2023-3-1/bin/pip -m venv /home/projects2/your_project_name/venvs/cpu/
```
Now press enter and wait for the environment to finishing building. 


### Sourcing the virtual environment
To activate / source the environment, type the following:
```bash
source /home/projects2/your_project_name/venvs/cpu/bin/activate
```
Now the environment is activated.

To deactivate the environment, write `deactivate` in the shell.


## Installing project and dependencies (workaround)
Here is a workaround to create the environment and install the package needed for the project while using the Health Tech Clusters' outdated Python version (3.10):

Make sure to update the pip and setuptools in the venv, and remember that Python 3.10 (which is the current version of Python on the server) only works with setuptools <=81.0
```bash
python3 -m pip install --upgrade pip 'setuptools<=81.0'
```

To install the necessary dependencies to run the models, simply type:

```bash
pip install scikit-learn matplotlib esm seaborn pandas argparse tqdm biopython joblib httpx fastparquet 'sourmash<=4.3' shap fastcluster scipy 'screed<=1.1.2'
pip install --no-deps .
```


# Training and evaluating the models

## Location of the data
Firstly, create a location to put the genomes and the metadata file containing the labels. 
```bash
mkdir /home/projects2/your_project_name/data/
mkdir /home/projects2/your_project_name/data/genomes
```
Also create a directory to store the results:
```bash
mkdir /home/projects2/your_project_name/results/
```

Next, we want to copy the data into the genomes folder

As an example, let's make a copy of the E.coli genomes for the gentamicin resistance task. 
```bash
cp /home/projects2/ecoli_gentamicin/all_genomes/* /home/projects2/your_project_name/data/genomes/
```

Let's also copy the metadata csv file to the directory
```bash
cp /home/projects2/ecoli_gentamicin/metadata_ecoli_gentamicin.csv /home/projects2/your_project_name/data/
```

## Running script in an interactive node
Make sure you have sourced the virtual environment first (see above). If you do not want to save the best model, remove the --save_best_model parameter.
```
python3 src/utilities/baselines.py  \
  --phenotype  resistant_phenotype \
  --input /home/projects2/your_project_name/data/genomes/ \
  --labels /home/projects2/your_project_name/data/metadata_ecoli_gentamicin.csv \
  --output /home/projects2/your_project_name/results/
  --model_directory /home/projects2/your_project_name/models/
  --kmer_prefix ATG \
  --kmer_suffix_size 8 \
  --id_column genome_id \
  --file_type fasta \
  --embedding frequency \
  --group_clusters \
  --submodule feature_importance \
  --model HistGradientBoosting \
  --save_model \
  
```

If you want to only train model (not run feature importance extraction), change --submodule to ´train´.


### Obs! To return to the login node, press `ctrl+d` from the terminal or write `exit` and press enter.


## Validation
When you have trained a model and you want to test the model on an external validation set, use the "validation" submodule. 
You should provide an existing model directory, using ´--model_directory´.

- Change the --input to point to the folder containing the validation genome files.
- Change the --labels to point to the validation csv.
- Change the --id_column to match column name of the column containing the genome ids (should be the same in the labels files and in the headers of the fasta files).

```
python3 src/utilities/baselines.py  \
  --phenotype  resistant_phenotype \
  --input /home/projects2/your_project_name/validation/genomes/ \
  --labels /home/projects2/your_project_name/validation/validation.csv \
  --output /home/projects2/your_project_name/results/
  --kmer_prefix ATG \
  --kmer_suffix_size 8 \
  --id_column genome_id \
  --file_type fasta \
  --embedding frequency \
  --group_clusters \
  --submodule validation \
  --model HistGradientBoosting \
  --model_directory /home/projects2/your_project_name/models/
```

## Sklearn Parallel (OpenMP)
As the HistGradientBoosting and RandomForest models use sklearns implementations, multicore can be enables using: 
```
OMP_NUM_THREADS=4 python3 src/utilities/baselines.py ...
```
See sklearns documentation: [\[Parallel\](https://scikit-learn.org/stable/computing/parallelism.html#parallel-numpy-and-scipy-routines-from-numerical-libraries)](https://scikit-learn.org/stable/computing/parallelism.html#lower-level-parallelism-with-openmp)

## Available options
To see all available options for the "baselines.py" write:
```
python3 src/utilities/baselines.py --help
```

## Creating and submitting SLURM script
Not written yet. Look in the shell_scripts/ folder to see some examples.

## Analysing the results

### Download the results from the server to a folder on your laptop:

If you don't already have a project folder on your local machine, now is a good time to create it.
```
cd ~/
mkdir your_local_project/
```

In the local terminal (not on the cluster) type this to copy the results to a local folder
```
mkdir your_local_project/results/
rsync s123456@login.healthtech.dtu.dk:/home/projects2/your_project_name/results/\* your_local_project/results/
```

Locally, open this notebook to inspect the results:
[Results notebook](../notebooks/results_analysis_notebook.ipynb)

# Examples for how to use the virtual environment
Source / activate the environment (see above), then the pip and python commands shown below will use your virtual environment.

#### Installing a library in the environment using pip:
```bash
pip install some_python_package
```

#### Running Python with your environment
```bash
python3 some_cool_python_program.py
```

