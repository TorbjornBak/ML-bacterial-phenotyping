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



## Installing dependencies 
To install the necessary dependencies to run the models, simply type: 
```bash
/home/projects2/your_project_name/venvs/cpu/bin/pip install -e .
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
```
/home/projects2/your_project_name/venvs/cpu/bin/python3 src/utilities/baselines.py  \
  --phenotype  resistant_phenotype \
  --input /home/projects2/your_project_name/data/genomes/ \
  --output /home/projects2/your_project_name/results/
  --labels /home/projects2/your_project_name/data/metadata_ecoli_gentamicin.csv \
  --kmer_prefix ACATG \
  --kmer_suffix_size 6 \
  --id_column genome_id \
  --dna_sequence_column dna_sequence \
  --file_type fasta \
  --embedding frequency \
  --group_clusters
```

### Obs! To return to the login node, press `ctrl+d` from the terminal or write `exit` and press enter.



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
rsync s123456@login.healthtech.dtu.dk:/home/projects2/your_project_name/results/* your_local_project/results/
```



# Examples for how to use the virtual environment

#### Installing a library in the environment using pip:
```bash
/home/projects2/your_project_name/venvs/cpu/bin/pip install some_python_package
```

#### Running Python with your environment
```bash
/home/projects2/your_project_name/venvs/cpu/bin/python3 some_cool_python_program.py
```
