
from utilities.cliargparser import ArgParser


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import umap.plot
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix

from embeddings import kmerize_joblib, load_labels
from dataclasses import dataclass

def load_stored_embeddings(dataset_file_path):
	print(f"Loading embeddings from: {dataset_file_path=}")
	z = np.load(dataset_file_path, allow_pickle=True)

	X = list(z["X"])  # object array → list of arrays 
	ids = list(z["ids"])  # map labels from current dict
	print(f'{len(X)=}')
	print(f'{len(ids)=}')
	return X, ids


def embed_data(label_dict, dir_list, kmer_prefix="CGTCA", kmer_suffix_size = 4, cores = 4, output_directory = None, output_type = "counts", reembed = False):

	dataset_name = f'{kmer_prefix}_{kmer_suffix_size}_{output_type}' 
	dataset_file_path = f'{output_directory}/{dataset_name}.npz'
	
	if not os.path.isfile(dataset_file_path) or reembed:
		result_dict = kmerize_joblib(dir_list, kmer_prefix=kmer_prefix, kmer_suffix_size=kmer_suffix_size, nr_of_cores=cores, output_type=output_type)
		data_dict = result_dict["joblib_result"]
		ids = [gid for gid in data_dict.keys()]
		X = [data_dict[gid] for gid in ids]

		np.savez_compressed(dataset_file_path, X=X, ids=np.array(ids, dtype=object))	
		print(f"{dataset_file_path=}")
	else:
		X, ids = load_stored_embeddings(dataset_file_path)
		

	X = [x for gid, x in zip(ids, X) if gid in label_dict]
	y = np.array([label_dict[gid] for gid in ids if gid in label_dict], dtype=np.int64)

	return X, y

def pca_plot(context):
	pca = PCA(n_components=2, random_state=0)
	X_pcs = pca.fit_transform(context.X)

	print(pca.explained_variance_ratio_)

	labels = np.unique(context.y)

	label2id = {label: i for i, label in enumerate(labels)}

	color_list = [label2id[l] for l in y]

	plt.figure(figsize=(6,5))

	plt.scatter(X_pcs[:, 0], X_pcs[:, 1], c=color_list, cmap='coolwarm', edgecolor='k')
	plt.xlabel(f'PC1')
	plt.ylabel(f'PC2')
	plt.title('PCA projection')
	plt.legend(title='Label', frameon=False)
	plt.tight_layout()
	pca_save_path = f'{context.output_directory}/pca_analysis_{context.phenotype}_prefix_{context.kmer_prefix}_suffix_size_{context.kmer_suffix_size}.jpg'
	plt.savefig(pca_save_path)

	print(f'{pca_save_path=}')


def pca_classification():
	pass

def random_forest_classification(context):
	# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

	X, y = context.X, context.y
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size= 0.2)
	clf = RandomForestClassifier(max_depth=10, random_state=0)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	#y_pred_probabilities = clf.predict_proba(X_test)
	print(f'{y_test[:100]=}')
	print(f'{y_pred[:100]=}')
	print(f'accuracy of random forest = {clf.score(X_test, y_test)}')
	
	create_classification_report(y_train, y_test, y_pred, context)

	return y_pred
	
def hist_gradient_boosting_classifier(context):
	# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
	X, y = context.X, context.y
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size= 0.2)
	#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = 42, test_size=1/8) 
						

	clf = HistGradientBoostingClassifier(loss = 'log_loss', 
									  learning_rate=0.01, 
									  l2_regularization = 1e-5,
									  max_features=0.9,
									  class_weight="balanced")
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	print(f'{y_test[:100]=}')
	print(f'{y_pred[:100]=}')
	print(f'accuracy of random forest = {clf.score(X_test, y_test)}')
	
	create_classification_report(y_train, y_test, y_pred, context)

	return y_pred


def umap_plot(context):
	mapper = umap.UMAP().fit(context.X)
	ax = umap.plot.points(mapper, labels = context.y)
	umap_save_path = f'{context.output_directory}/umap_{context.phenotype}_prefix_{context.kmer_prefix}_suffix_size_{context.kmer_suffix_size}.png'
	ax.figure.savefig(umap_save_path)
	print(f'{umap_save_path=}')


@dataclass
class model_context:
	X: np.array
	y: np.array
	output_directory: str
	phenotype: str
	kmer_prefix: str
	kmer_suffix_size: int
	model_type: str
	int2label: dict

def create_classification_report(y_train, 
								 y_test, 
								 y_pred, 
								 ctx):

	report = classification_report(y_test, y_pred, output_dict=True, zero_division="warn")
	conf_matrix = confusion_matrix(y_test, y_pred, labels = list(ctx.int2label))

	#y_test_oh = np.eye(len(np.unique(y_train)))[y_test]
	#auc_weighted = roc_auc_score(y_test_oh, y_pred_probabilities, average="weighted", multi_class="ovr")
	#auc_macro = roc_auc_score(y_test_oh, y_pred_probabilities, average="macro", multi_class="ovr")

	# Calculate balanced accuracy
	balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

	# Store results
	results = pd.Series(
		{
			"phenotype": ctx.phenotype,
			"model_name": ctx.model_type,
			"kmer_prefix": ctx.kmer_prefix,
			"kmer_suffix_size": ctx.kmer_suffix_size,
			"f1_score_weighted": report["weighted avg"]["f1-score"],
			"f1_score_macro": report["macro avg"]["f1-score"],
			"precision_weighted": report["weighted avg"]["precision"],
			"precision_macro": report["macro avg"]["precision"],
			"recall_weighted": report["weighted avg"]["recall"],
			"recall_macro": report["macro avg"]["recall"],
			"accuracy": report["accuracy"],
			"balanced_accuracy": balanced_accuracy,
			#"auc_weighted": auc_weighted,
			#"auc_macro": auc_macro,
			"n_classes": len(np.unique(y_train)),
			"confusion_matrix" : conf_matrix,
			"intx2label" : ctx.int2label,
		}
		)
	dataset_name = f"tmp_result_{ctx.model_type}_{ctx.phenotype}_{ctx.kmer_prefix}_{ctx.kmer_suffix_size}"
	path = f'{ctx.output_directory}/{dataset_name}.csv'
	results.to_csv(path)
	print(f'Saved tmp result to {path=}')
	print(f'{results=}')
	return results



if __name__ == "__main__":

	
	parser = ArgParser(module = "pca_analysis")
	parser = parser.parser
	
	
	phenotypes = parser.phenotype # is a list 
 
	labels_path = parser.labels_path
	id_column = parser.id_column
	input_data_directory = parser.input
	output_data_directory = parser.output
	reembed = parser.reembed

	for phenotype in phenotypes:

		label_return = load_labels(file_path=labels_path, id = id_column, label = phenotype, sep = ",")
		label_dict_literal, label_dict, int2label = label_return["label_dict"], label_return["label_dict_int"], label_return["int2label"] 

		file_suffix = ".parquet"
		dir_list = os.listdir(input_data_directory)
		dir_list = [f'{input_data_directory}/{file}' for file in dir_list if file_suffix in file]

		print(f'{dir_list=}')

		#kmer_prefix = cli_arguments["--KMER"] if "--KMER" in cli_arguments else "CGTCA"
		kmer_prefix = parser.kmer_prefix
		#kmer_suffix_size = int(cli_arguments["--SUFFIX_SIZE"]) if "--SUFFIX_SIZE" in cli_arguments else 4
		kmer_suffix_size = parser.kmer_suffix_size
		
		X, y = embed_data(label_dict=label_dict, 
					dir_list=dir_list, 
					kmer_prefix=kmer_prefix, 
					kmer_suffix_size = kmer_suffix_size, 
					cores = parser.cores, 
					output_directory = output_data_directory, 
					output_type = "counts",
					reembed=reembed
					)


		ctx = model_context(
							X,
							y, 
							output_data_directory, 
							phenotype, 
							kmer_prefix, 
							kmer_suffix_size,
							model_type="RandomForest",
							int2label=int2label)
		
		

		y_pred = random_forest_classification(ctx)

		ctx.model_type = "HistGradientBoosting" 

		y_pred = hist_gradient_boosting_classifier(ctx)


		# Plotting pca and umap
		pca_plot(ctx)
		umap_plot(ctx)

		