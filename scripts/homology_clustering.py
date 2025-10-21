from kmer_sampling import kmerize_parquet_joblib, load_labels
import numpy as np
import sys, os
import pandas as pd

import seaborn as sns



def load_stored_embeddings(dataset_file_path):
    print(f"Loading embeddings from: {dataset_file_path=}")
    z = np.load(dataset_file_path, allow_pickle=True)

    X = list(z["X"])  # object array → list of arrays 
    ids = list(z["ids"])  # map labels from current dict
    return X, ids




#X, y = embed_data(prefix=prefix, suffix_size=suffix_size, input_data_directory=input_data_directory)


def embed_data(label_dict, dir_list, path = None, kmer_prefix="CGTCA", kmer_suffix_size = 4, cores = 4, output_type = "bytearray", store = True):
    if store:
        data_dict = kmerize_parquet_joblib(dir_list, kmer_prefix=kmer_prefix, kmer_suffix_size=kmer_suffix_size, nr_of_cores=cores, output_type=output_type)
        ids = [gid for gid in data_dict.keys()]
        X = [data_dict[gid] for gid in ids]

        
        dataset_name = f'{output_type}_{kmer_prefix}_{kmer_suffix_size}' 
        dataset_file_path = f'{path}/{dataset_name}.npz'
        print(f"Saving embeddings to: {dataset_file_path=}")
        np.savez_compressed(dataset_file_path, X=X, ids=np.array(ids, dtype=object))

    else:
        dataset_name = f'{output_type}_{kmer_prefix}_{kmer_suffix_size}' 
        dataset_file_path = f'{path}/{dataset_name}.npz'
        X, ids = load_stored_embeddings(dataset_file_path=dataset_file_path)


    X = [x for gid, x in zip(ids, X) if gid in label_dict]
    ids = [gid for gid in ids if gid in label_dict]
    y = np.array([label_dict[gid] for gid in ids if gid in label_dict], dtype=np.int64)

    return X, y, ids


def jaccard_similarity(A, B):
    # Input two arrays
    intersect = sum(A & B)

    union = sum(A | B)
    similarity = (intersect / union)
    #print(distance)
    return similarity

def clustering(arr, ids, cutoff = 0.03):
    clusters = list() # list of tuples: (id, kmers)
    clusters_dict = dict() # key: id, value = set of ids in the cluster
    
    print(f'{len(arr)=}')
    print(f'{len(ids)=}')

    # Contains id, label 
    for kmers, gid in zip(arr, ids):
        #print(gid)
        assign_cluster = False
        for cluster in clusters:
            if jaccard_similarity(kmers, cluster[1]) <= cutoff:
                cluster_id = cluster[0]
                clusters_dict[cluster_id].add(gid)

                assign_cluster = True

                break
        
        if assign_cluster == False:
            clusters.append((gid, kmers))
            clusters_dict[gid] = set({gid})
            #print(clusters[0])

    print(f'{len(clusters)=}')
    # clusters_dict is a dict of sets
    # clusters is a dict containing the kmers for the cluster parent
    return clusters, clusters_dict


def cluster_stats(clusters_dict, label_dict_literal):

    stats = dict()

    for gid, cluster_set in clusters_dict.items():
        stats[gid] = dict()
        for seq in cluster_set:
            label = label_dict_literal[seq]
            if label not in stats[gid]:
                stats[gid][label] = 1
            else:
                stats[gid][label] += 1
    
    return stats



def distance_matrix(X):
    print("Calculating distance matrix")
    dist_matrix = np.zeros((len(X), len(X)))
    i = 0
    for a in range(len(X)):
        j = i
        for b in range(len(X)):
            #print(arr1, arr2)
            arr1 = X[i]
            arr2 = X[j]
            dist = (jaccard_similarity(arr1,arr2))
            dist_matrix[j,i] = dist
            dist_matrix[i,j] = dist
            #print(f'{dist_matrix=}')
            #print(f'{i=}, {j=}')
            j += 1
            if j >= len(X):
                break
            #print(arr)
        i += 1
    return dist_matrix

def distance_matrix_old(X):
    print("Calculating distance matrix")
    dist_matrix = np.zeros((len(X), len(X)))
    i = 0
    for arr1 in X:
        j = 0
        for arr2 in X:
            #print(arr1, arr2)
     
            dist = (jaccard_similarity(arr1,arr2))
            dist_matrix[j,i] = dist
         
            print(f'{dist_matrix=}')
            print(f'{i=}, {j=}')
            j += 1
         
            #print(arr)
        i += 1
    return dist_matrix
        

if __name__ == "__main__":
    prefix = 'CGTC'
    suffix_size = 6

    phenotype = "madin_categorical_gram_stain"

    # labels_path = "/home/projects2/bact_pheno/bacbench_data/labels.csv"
    # input_data_directory = "/home/projects2/bact_pheno/bacbench_data"

    labels_path = "downloads/labels.csv"
    input_data_directory = "downloads"


    file_suffix = ".parquet"

    dir_list = os.listdir(input_data_directory)
    dir_list = [f'{input_data_directory}/{file}' for file in dir_list if file_suffix in file]
	
    label_dict_literal, label_dict = load_labels(file_path=labels_path, id = "genome_name", label = phenotype, sep = ",")

    X, y, ids = embed_data(label_dict=label_dict, 
                           path = "results/homology", 
                           dir_list=dir_list, 
                           kmer_prefix=prefix, 
                           kmer_suffix_size = suffix_size, 
                           cores = 3, 
                           output_type="bytearray", 
                           store=True)

    X = X
    ids = ids

    
    
    #print(arr)

    clusters, clusters_dict = clustering(X, ids, cutoff=0.2)

    print(cluster_stats(clusters_dict, label_dict_literal))

    kmers = [kmer for id, kmer in clusters]

    ids = [id for id, kmer in clusters]
    
    labels = [label_dict_literal[id] for id, kmer in clusters]

    
    arr = distance_matrix(kmers)
    df = pd.DataFrame(arr, columns=labels, index = labels)
        
    # https://seaborn.pydata.org/generated/seaborn.heatmap.html


    ax = sns.heatmap(df, annot = False, cmap="crest")
    #ax.set(xlabel="Kmer Suffix Size", ylabel="Kmer Prefix", title ="Balanced Accuracy")
    figure = ax.get_figure()
    figure.savefig("results/homology_1_4.jpg", dpi=400)