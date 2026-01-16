import os
import pandas as pd
import seaborn as sns
# Helper functions for plotting results from training runs

def get_files(path = "../results/training_CNN_v2_lr3/"):

    file_suffix = ".csv"
    dir_list = os.listdir(path)
    dir_list = [f'{path}/{file}' for file in dir_list if file_suffix in file]
    print(dir_list)
    print(f'{len(dir_list)=}')

    return dir_list


def read_series_csv(path):


    df = pd.read_csv(path, skiprows=1, header = None)                 # two columns: index, value [web:31]
    if df.shape[1] == 2:
        df = df.set_index(0).T                       # index to columns, values to a single row [web:63]
        df.columns.name = None
    
    return df



def format_df(df, 
                metric_cols = [
                "f1_score_weighted","f1_score_macro","precision_weighted","precision_macro",
                "recall_weighted","recall_macro","accuracy","balanced_accuracy","auc_weighted","auc_macro"
                ]):
    

    df[metric_cols] = df[metric_cols].apply(pd.to_numeric)

    df["kmer_suffix_size"] = df["kmer_suffix_size"].apply(pd.to_numeric)


    df["model_name"] = df["model_name"].astype(str).str.replace("_ONEHOT",""
                        ).str.replace("HistGradientBoosting","HGB"
                        ).str.replace("_MLP",""
                        ).str.replace("_ESMC",""
                        ).str.replace("_"," ")
    
        
    if "embedding_class" in df.columns:
        
        df["Parameters"] = (
        df["model_name"].astype(str) + ", " +
        df["embedding_class"].astype(str) + ", " +
        df["kmer_prefix"].astype(str) + ", " +
        df["kmer_suffix_size"].astype(str)
        )
    else:
        df["Parameters"] = (
        df["model_name"].astype(str) + ", " +
        df["kmer_prefix"].astype(str) + ", " +
        df["kmer_suffix_size"].astype(str)
        )


    return df

def get_avg(path, id_cols = ["phenotype","model_name","kmer_prefix","kmer_suffix_size"]):
    df = pd.concat((read_series_csv(path) for path in get_files(path)))

    metric_cols = [
        "f1_score_weighted","f1_score_macro","precision_weighted","precision_macro",
        "recall_weighted","recall_macro","accuracy","balanced_accuracy","auc_weighted","auc_macro"
    ]

    df = format_df(df, metric_cols)
    

    avg_by_config = (
        df
        .groupby(id_cols, as_index=False)[metric_cols]
        .mean()
    )
    return avg_by_config

def reformat_x_labels(df_column):
    df_column = df_column.str.replace("_"," ").str.replace("madin","").str.replace("gideon","").str.replace("categorical","").str.replace("Carbsubs","Utilization of").str.strip().str.title()
    
    return df_column


def confusion_plot(path):
    df = read_series_csv(path)
    
    df = df[["phenotype","model_name","confusion_matrix","f1_score_weighted","balanced_accuracy"]]

    #print(df)
   
    conf = df.confusion_matrix.to_list()[0].split("\n ")
    conf = [[int(i) for i in i.replace("[","]").replace("]","").split()]  for i in conf]
    
    ax = sns.heatmap(conf, annot = True, cmap="crest",fmt='g')
    df["phenotype"] = reformat_x_labels(df["phenotype"])
    ax.set(title = f"Confusion matrix for {df['model_name'].iloc[0]} - task: {df['phenotype'].iloc[0]}")

    print(f'{float(df.iloc[0]["balanced_accuracy"])=}, {conf=}')
    return ax