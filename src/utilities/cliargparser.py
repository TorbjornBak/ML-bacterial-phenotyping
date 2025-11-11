# Idea is to make a class that returns the different arguments parse in the cli to the different programs, 
# to unify the inputs across my different tools


import argparse, os

class ArgParser():

    def __init__(self, module = None):
   
        if module == "train_models":
            parser = self.train_model_arguments()
        elif module == "pca_analysis":
            parser = self.pca_arguments()
        elif module is not None:
            raise ValueError(f"Unknown module '{module}'")
                             
        self.parser = parser.parse_args()
        
        self.check_inputs()

    def check_inputs(self):
        parser = self.parser
        labels_path = parser.labels_path
        parser.input = parser.input.rstrip("/")
        parser.output = parser.output.rstrip("/")
        input_data_directory = parser.input
        output_data_directory = parser.output
        assert os.path.isdir(output_data_directory), f"Selected output directory does not exist: {output_data_directory}"
        assert os.path.isdir(input_data_directory), f"Selected input directory does not exist: {input_data_directory}"
        assert os.path.isfile(labels_path), f"Path to labels does not exist: {labels_path}"
        
        self.parser = parser

    def default_arguments(self, parser):
        parser.add_argument("--phenotype", type = str, nargs = "+", help = "Phenotype - target from file", required = True)
        parser.add_argument("--cores", default=2, type = int, help="nr of cores to use for embedding")
        parser.add_argument("--input", required = True, help = "Path to input directory containing files for training")
        parser.add_argument("--labels_path", required = True, help = "Path to file containing labels for training")
        parser.add_argument("--id_column", default = "genome_name", type = str, help = "Name of column containing ids for labels")
        parser.add_argument("--dna_sequence_column", required = True, help = "Name of column containing dna sequences in parquet files")
        parser.add_argument("--output", required = True, help = "Path to output directory for training results")
        parser.add_argument("--file_type", default = "parquet", help = "fx parquet / fasta, the file ending to look for in input folder")
        parser.add_argument("--freq_others", default = None, help = "Bottom frequency of label counts that should be discarded. Useful if some labels only occur a few times.")
        parser.add_argument("--reembed", default = False, action="store_true", help = "Flag determining whether to run tokenization and embedding again")
        return parser
    
    def train_model_arguments(self):
        parser = argparse.ArgumentParser(
                                        prog='train_models.py',
                                        description='Toolbox of different ml models for downsampled genomes',
                                        epilog='Made by Torbjørn Regueira',
                                        )
        
        parser = self.default_arguments(parser)
        parser.add_argument("--model_arch", default="CNN", help = "Determines which ml model architecure to use, (CNN, RNN or TRANSFORMER)")
        parser.add_argument("--lr", "--learning_rates", default = -1.0, type = float, nargs = '+', help = "List of learning rates for given model")
        parser.add_argument("--kmer_prefixes", required = True, type = str, nargs = '+', help = "Comma separated list of kmer prefixes")
        parser.add_argument("--kmer_suffix_sizes", required = True, type = int, nargs = '+', help = "Comma seaparated list of kmer suffix sizes")
        parser.add_argument("--compress", action="store_true", help = "Flag telling whether to compress vocab size or not")
        parser.add_argument("--embed_only", action = "store_true", help = "Flag to tell whether to only embed, not train")
        parser.add_argument("--trace_memory", action = "store_true", help = "Flag to tell whether to trace memory usage")
        parser.add_argument("--epochs", default = 150, type = int, help = "Nr of epochs to train for, for each model")
        parser.add_argument("--dropout", default = 0.2, type = float, help = "%% to dropout for each layer")
        parser.add_argument("--k_folds", default = 3, type = int, help = "Nr of folds for cross validation")
        parser.add_argument("--patience", default = 15, type = int, help = "Parameter to determine how long to wait before early stopping")
        parser.add_argument("--tokenize_method", default="kmers", type = str, help = "Method for tokenization, choose between kmers and bpe" )
        parser.add_argument("--embedding", default = "integer", help = "Type of embedding, choose between integer and esmc")
        parser.add_argument("--esmc_model", default = "esmc_300m", help = "Which ESM-c model to use for embedding, fx esmc_300m or esmc_1b")
        parser.add_argument("--pooling", default = "mean_per_token", help = "Pooling method for ESM-c embeddings, choose between mean, mean_per_token or None")
        return parser
        
    def pca_arguments(self):
        parser = argparse.ArgumentParser(
                                        prog='src/utilities/pca_analysis.py',
                                        description='Toolbox of different ml models for downsampled genomes',
                                        epilog='Made by Torbjørn Regueira',
                                        )
        parser = self.default_arguments(parser)
        parser.add_argument("--kmer_prefix", required = True, help = "Kmer prefix to use for pca and umap")
        parser.add_argument("--kmer_suffix_size", required = True, type = int, help = "Kmer suffix size to use for pca and umap")
        
        return parser


   

        