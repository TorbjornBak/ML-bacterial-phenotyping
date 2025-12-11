import matplotlib.pyplot as plt
import seaborn as sns

import os
import torch
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, train_test_split, GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

from embeddings.KmerTokenization import KmerTokenizer, load_labels
from embeddings.integer_embeddings import KmerCountsEmbeddings
from embeddings.esmc_embeddings import ESMcEmbeddings
from utilities.cliargparser import ArgParser
from utilities.clustering import SourMashClustering

import shap
from dataclasses import dataclass

def is_embedding_file(dataset_file_path, embedding_class = "frequency"):

	if embedding_class == "frequency":
		file_types = [".npz"]
	elif embedding_class == "counts":
		file_types = [".npz"]
	elif embedding_class == "esmc":
		file_types = [".npz", ".pt"]

	else:
		raise ValueError(f"Embedding class {embedding_class} not recognized. Aborting...")
	
	print(f'Checking for embedding file at: {dataset_file_path} with types: {file_types}')

	for type in file_types:
		if not os.path.isfile(f'{dataset_file_path}{type}'):
			print(f'Embedding file not found: {dataset_file_path}{type}')
			return False
	print(f'Embedding files found.')
	return True


def embed_data(label_dict, 
			   input_data_directory, 
			   output_data_directory,
			   kmer_prefix=None, 
			   kmer_suffix_size = None, 
			   kmer_offset = 0,
			   id_column = "genome_name", 
			   sequence_column = "dna_sequence", 
			   embedding_class = "frequency",
			   cores = 4, 
			   file_type = "parquet", 
			   reembed = False, 
			   reverse_complement = False,
			   esmc_model = "esmc_300m",
			   esmc_pooling = "mean",
			   device = "cpu",
			   group_clusters = False):

	if embedding_class in ["frequency", "counts"]:

		embedder = KmerCountsEmbeddings(
						kmer_prefix=kmer_prefix,
						kmer_suffix_size=kmer_suffix_size,
						kmer_offset=kmer_offset,
						data_directory=output_data_directory,
						embedding_class=embedding_class,
						grouped=group_clusters,
		)
	elif embedding_class == "esmc":
		embedder = ESMcEmbeddings(
						kmer_prefix=kmer_prefix,
						kmer_suffix_size=kmer_suffix_size,
						kmer_offset=kmer_offset,
						data_directory=output_data_directory,
						esmc_model=esmc_model,
						pooling=esmc_pooling,
						device=device,
						grouped=group_clusters,
		)
	else:
		raise ValueError(f"Embedding class {embedding_class} not recognized. Aborting...")

	if reembed or not is_embedding_file(embedder.file_path, embedding_class=embedder.embedding_class):
		
		tokenizer = KmerTokenizer(
							input_data_directory,
							genome_col=id_column,
							dna_sequence_col=sequence_column,
							kmer_prefix=kmer_prefix,
							kmer_suffix_size=kmer_suffix_size,
							file_type=file_type,
							reverse_complement=reverse_complement,
							kmer_offset = kmer_offset,
							)
		token_collection = tokenizer.run_tokenizer(nr_of_cores=cores)

		embeddings = embedder.run_embedder(token_collection=token_collection)

		gid_and_strand_id = [[gid, strand_id] for gid, strands in embeddings.items() for strand_id in strands]

		X = [embeddings[gid][strand_id] for gid, strand_id in gid_and_strand_id]
		strand_ids = [strand_id for _, strand_id in gid_and_strand_id] # All strand ids (forwards, reverse)
		genome_ids = [gid for gid, _ in gid_and_strand_id] # Genome ids
		print(f'{len(np.unique(genome_ids))=}')
		print(f'{len(genome_ids)=}')

		# Create groups based on clustering
		if group_clusters:
			print(f'Grouping clusters to avoid data leakage during train test split...')
			clusterer = SourMashClustering(kmer_suffix_size=kmer_suffix_size, target_labels=None, n = 1000)
			minhashes = clusterer.hash_tokens(token_dict=token_collection)
			distance_matrix, labels = clusterer.jaccard_distance_matrix(minhashes=minhashes)
			cluster_groups = clusterer.group_clusters(distance_matrix=distance_matrix, labels=labels, method = "average", nr_of_clusters=40)
			print(f'{np.unique(cluster_groups)=}')

			# Merge with groups

			# Join gene id (group) with cluster group, both forward and reverse strand should have same cluster group
			step = len(genome_ids) // len(np.unique(genome_ids)) # Should be 1 or 2
			if step > 1:
				combined_groups = []
				assert step in [2], f"step should be one of 1 or 2, was {step}"
				for i in range(0, len(genome_ids), step):
					for _ in range(step):
						combined_groups.append(cluster_groups[i])

				groups = np.array(combined_groups)

			else:
				groups = np.array(cluster_groups)
		else:
			groups = np.array(genome_ids)
			

		assert len(X) == len(strand_ids) == len(groups), "Length mismatch in embeddings output!"
		assert len(X) > 0, "No embeddings were created! Aborting..."
		print(f'{len(X)=}')
		print(f'{len(strand_ids)=}')
		print(f'{len(groups)=}')

		embedder.save_embeddings(X, strand_ids, groups, genome_ids)
	else:
		X, strand_ids, groups, genome_ids, channel_size = embedder.load_stored_embeddings()
		
	if embedder.embedding_class == "esmc":
		if esmc_pooling == "mean":
			X = np.array(
				[
					(x.detach().cpu() if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32))
					for gid, x in zip(genome_ids, X) if gid in label_dict
				],
				dtype=np.float32
			)
			
			if X.ndim == 3 and X.shape[1] == 1:
				X = X[:, 0, :]      # (962, 960)
		else:
			raise NotImplementedError(f"Pooling method {esmc_pooling} not implemented for loading embeddings.")
		
	else:
		X = np.array([x for gid, x in zip(genome_ids, X) if gid in label_dict], dtype = object)

	y = np.array([label_dict[gid] for gid in genome_ids if gid in label_dict])
	groups = np.array([group_id for group_id, gid in zip(groups, genome_ids) if gid in label_dict])
	
	print(f'{len(X)=}')
	print(f'{len(y)=}')
	assert len(X) == len(y), "Length mismatch between X and y!"

	return X, y, groups

def random_forest_classification(context):
	# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
	context.model_type = "RandomForest"

	if context.train_split_method == "GroupShuffleSplit":
		splitter = GroupShuffleSplit(n_splits = context.k_folds, test_size = 0.2, random_state=42)
	elif context.train_split_method == "GroupKFold":
		splitter = GroupKFold(n_splits = context.k_folds, random_state = 42, shuffle = True)
	else:
		raise ValueError(f"Train split method {context.train_split_method} not recognized.")
	i = 0
	for train, test in splitter.split(context.X, context.y, groups = context.groups):
		X_train, y_train = context.X[train], context.y[train]
		X_test, y_test = context.X[test], context.y[test]

		clf = RandomForestClassifier(max_depth=None, 
							   		random_state=0)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)

		print(f'{y_test[:100]=}')
		print(f'{y_pred[:100]=}')
		print(f'Accuracy of RandomForest: {clf.score(X_test, y_test)}')
		
		create_classification_report(y_train=y_train, 
							   y_test=y_test, 
							   y_pred=y_pred, 
							   seed=i, 
							   ctx=context)
		i+= 1
	print(f'Finished RandomForest classification over {context.k_folds} folds.')
	return

	for seed in range(context.k_folds):
	
		X_train, X_test, y_train, y_test = train_test_split(context.X, context.y, random_state = seed, test_size= 0.2)
		clf = RandomForestClassifier(max_depth=10, 
							   		random_state=0)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		
		print(f'{y_test[:100]=}')
		print(f'{y_pred[:100]=}')
		print(f'Accuracy of RandomForest: {clf.score(X_test, y_test)}')
		
		create_classification_report(y_train=y_train, 
							   y_test=y_test, 
							   y_pred=y_pred, 
							   seed=seed, 
							   ctx=context)
		
	print(f'Finished RandomForest classification over {context.k_folds} folds.')
	return clf

	
def hist_gradient_boosting_classifier(context):
	# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
	clf = None
	context.model_type = "HistGradientBoosting"
	print(f'Running HistGradientBoostingClassifier for classification...')
	if context.train_split_method == "GroupShuffleSplit":
		splitter = GroupShuffleSplit(n_splits = context.k_folds, test_size = 0.2, random_state=42)
	elif context.train_split_method == "GroupKFold":
		splitter = GroupKFold(n_splits = context.k_folds, random_state = 42, shuffle = True)
	else:
		raise ValueError(f"Train split method {context.train_split_method} not recognized.")
	i=0
	for train, test in splitter.split(context.X, context.y, groups = context.groups):
		X_train, y_train = context.X[train], context.y[train]
		X_test, y_test = context.X[test], context.y[test]

		clf = HistGradientBoostingClassifier(
										loss = 'log_loss', 
										learning_rate=0.01, 
										l2_regularization = 1e-3,
										max_features=0.9,
										class_weight="balanced"
										)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)

		print(f'{y_test[:100]=}')
		print(f'{y_pred[:100]=}')
		print(f'Accuracy: {clf.score(X_test, y_test)}')
		
		create_classification_report(y_train=y_train, 
							   y_test=y_test, 
							   y_pred=y_pred, 
							   seed=i, 
							   ctx=context)
		i+=1
	print(f'Finished HistGradientBoosting classification over {context.k_folds} splits.')
	return 
	for seed in range(context.k_folds):
		X_train, X_test, y_train, y_test = train_test_split(context.X, context.y, random_state = seed, test_size= 0.2)
		clf = HistGradientBoostingClassifier(
										loss = 'log_loss', 
										learning_rate=0.01, 
										l2_regularization = 1e-3,
										max_features=0.9,
										class_weight="balanced"
										)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)


		# print(f'{y_test[:100]=}')
		# print(f'{y_pred[:100]=}')
		print(f'Accuracy of HistGradientBoost: {clf.score(X_test, y_test)}')
		
		create_classification_report(y_train=y_train, 
							   y_test=y_test, 
							   y_pred=y_pred, 
							   seed=seed, 
							   ctx=context)
	
	
	return clf




def feature_importance_extraction(context):
	# Used for feature importance extraction
	# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
	#models = []
	
	context.model_type = "HistGradientBoosting"

	print(f'Running HistGradientBoostingClassifier for feature importance extraction...')
	if context.train_split_method == "GroupShuffleSplit":
		splitter = GroupShuffleSplit(n_splits = context.k_folds, test_size = 0.2, random_state=42)
	elif context.train_split_method == "GroupKFold":
		splitter = GroupKFold(n_splits = context.k_folds, random_state = 42, shuffle = True)
	else:
		raise ValueError(f"Train split method {context.train_split_method} not recognized.")
	
	i = 0
	for train, test in splitter.split(context.X, context.y, groups = context.groups):
		X_train, y_train = context.X[train], context.y[train]
		X_test, y_test = context.X[test], context.y[test]

		clf = HistGradientBoostingClassifier(
										loss = 'log_loss', 
										learning_rate=0.01, 
										l2_regularization = 1e-3,
										max_features=0.9,
										class_weight="balanced"
										)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)

		print(f'{y_test[:100]=}')
		print(f'{y_pred[:100]=}')
		print(f'Accuracy: {clf.score(X_test, y_test)}')
		
		create_classification_report(y_train=y_train, 
							   y_test=y_test, 
							   y_pred=y_pred, 
							   seed=i, 
							   ctx=context)
		feature_names = [f'{context.kmer_prefix}{integer_to_kmer(j, context.kmer_suffix_size)}' for j in range(len(context.X[0]))]
		shap_values = get_shap_values(clf, pd.DataFrame(X_test, columns = feature_names)) # Convert to dataframe for feature names on the plots
		plot_shap_summary(shap_values, context, i)
		i+=1
	
	
	print(f'Finished HistGradientBoosting classification over {context.k_folds} folds.')
	
	return clf

	for seed in range(context.k_folds):
		X_train, X_test, y_train, y_test = train_test_split(context.X, context.y, random_state = seed, test_size= 0.2)
		clf = HistGradientBoostingClassifier(
										loss = 'log_loss', 
										learning_rate=0.01, 
										l2_regularization = 1e-3,
										max_features=0.9,
										class_weight="balanced"
										)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)

		
		# print(f'{y_test[:100]=}')
		# print(f'{y_pred[:100]=}')
		print(f'Accuracy of GradientBoosting: {clf.score(X_test, y_test)}')
		
		create_classification_report(y_train=y_train, 
							   y_test=y_test, 
							   y_pred=y_pred, 
							   seed=seed, 
							   ctx=context)
		
		feature_names = [f'{context.kmer_prefix}{integer_to_kmer(i, context.kmer_suffix_size)}' for i in range(len(context.X[0]))]
		# result = permutation_importance(
		# 	clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
		# )
		# 
		# forest_importances = pd.Series(result.importances_mean, index=feature_names)
		# print(f'{forest_importances.nlargest(10)=}')
		# forest_importances.to_csv(f'{context.output_directory}/feature_importances_{context.embedding_class}_{context.phenotype}_prefix_{context.kmer_prefix}_suffix_size_{context.kmer_suffix_size}_seed_{seed}.csv')
		#models.append(clf)
		shap_values = get_shap_values(clf, pd.DataFrame(X_test, columns = feature_names)) # Convert to dataframe for feature names on the plots
		plot_shap_summary(shap_values, context, seed)

	print(f'Finished GradientBoosting classification over {context.k_folds} folds.')
	#return models


def get_shap_values(model, X):
	#explainer = shap.explainers.Permutation(model, X)
	explainer = shap.TreeExplainer(model)
	shap_values = explainer(X)
	return shap_values

def plot_shap_summary(shap_values, context, seed):
	shap.plots.bar(shap_values, show = False)
	path = f'{context.output_directory}/shap_bar_{context.embedding_class}_{context.phenotype}_prefix_{context.kmer_prefix}_suffix_size_{context.kmer_suffix_size}_seed_{seed}.png'
	plt.title('SHAP Feature Importance')
	plt.savefig(path, bbox_inches='tight', dpi=300)
	plt.close()
	print(f'Saved SHAP bar plot to: {path}')

	# shap.plots.bar(shap_dict, show = False)
	# path = f'{context.output_directory}/shap_bar_divided_{context.embedding_class}_{context.phenotype}_prefix_{context.kmer_prefix}_suffix_size_{context.kmer_suffix_size}_seed_{seed}.png'
	# plt.savefig(path)
	# plt.close()
	# print(f'Saved SHAP bar plot to: {path}')

	shap.plots.beeswarm(shap_values, order=shap_values.abs.max(0), show = False)
	path = f'{context.output_directory}/shap_beeswarm_{context.embedding_class}_{context.phenotype}_prefix_{context.kmer_prefix}_suffix_size_{context.kmer_suffix_size}_seed_{seed}.png'
	plt.savefig(path, bbox_inches='tight', dpi=300)
	plt.close()
	print(f'Saved SHAP beeswarm plot to: {path}')

	# shap.force_plot(expected_value, y_pred, X_test, matplotlib=True)
	# path = f'{context.output_directory}/shap_force_{context.embedding_class}_{context.phenotype}_prefix_{context.kmer_prefix}_suffix_size_{context.kmer_suffix_size}_seed_{seed}.png'
	# plt.savefig(path)
	# plt.close()
	# print(f'Saved SHAP force plot to: {path}')

	# shap.plots.violin(shap_values, features=X_test, feature_names=feature_names, plot_type="layered_violin", show = False)
	# path = f'{context.output_directory}/shap_violin_{context.embedding_class}_{context.phenotype}_prefix_{context.kmer_prefix}_suffix_size_{context.kmer_suffix_size}_seed_{seed}.png'
	# plt.savefig(path)
	# plt.close()
	# print(f'Saved SHAP violin plot to: {path}')
	# shap.plots.heatmap(shap_values)
	# path = f'{context.output_directory}/shap_heatmap_{context.embedding_class}_{context.phenotype}_prefix_{context.kmer_prefix}_suffix_size_{context.kmer_suffix_size}_seed_{seed}.png'
	# plt.savefig(path)
	# print(f'Saved SHAP heatmap plot to: {path}')




def integer_to_kmer(x: int, k: int) -> str:
		inv = 'ACGT'
		out = []
		for _ in range(k):
			out.append(inv[x & 3])  # x % 4
			x >>= 2                 # x //= 4
		kmer = ''.join(reversed(out))
		return kmer

def pca_plot(context, save = True):
	pca = PCA(n_components=2, random_state=0)
	X_pcs = pca.fit_transform(context.X)

	print(pca.explained_variance_ratio_)

	labels = np.unique(context.y)

	label2id = {label: i for i, label in enumerate(labels)}

	color_list = [label2id[l] for l in context.y]

	plt.figure(figsize=(6,5))

	sns.scatterplot(x=X_pcs[:, 0], y=X_pcs[:, 1], hue=color_list,)
	plt.xlabel(f'PC1')
	plt.ylabel(f'PC2')
	plt.title('PCA projection')
	plt.legend(title='Label', frameon=False)
	plt.tight_layout()

	if save:
		pca_save_path = f'{context.output_directory}/pca_{context.embedding_class}_{context.phenotype}_prefix_{context.kmer_prefix}_suffix_size_{context.kmer_suffix_size}.png'
		plt.savefig(pca_save_path)

		print(f'{pca_save_path=}')
	plt.show()



# def umap_plot(context):
# 	mapper = umap.UMAP().fit(context.X)
# 	ax = umap.plot.points(mapper, labels = context.y)
# 	umap_save_path = f'{context.output_directory}/umap_{context.phenotype}_prefix_{context.kmer_prefix}_suffix_size_{context.kmer_suffix_size}.png'
# 	ax.figure.savefig(umap_save_path)
# 	print(f'{umap_save_path=}')



@dataclass
class model_context:
	X: np.array
	y: np.array
	groups: np.array
	grouped: bool
	output_directory: str
	phenotype: str
	kmer_prefix: str
	kmer_suffix_size: int
	model_type: str
	int2label: dict
	k_folds: int
	embedding_class: str	
	train_split_method: str = "GroupShuffleSplit"  # or GroupKFold

def create_classification_report(y_train,
								 y_test, 
								 y_pred, 
								 seed,
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
			"seed" : seed,
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
			"n_classes": len(np.unique(y_train)),
			"confusion_matrix" : conf_matrix,
			"int2label" : ctx.int2label,
			"grouped": ctx.grouped,
			"groups" : len(set(ctx.groups)),
			"embedding_class": ctx.embedding_class,
		}
		)
	

	dataset_name = f"tmp_result_{ctx.embedding_class}_{ctx.model_type}_{ctx.phenotype}_{"grouped" if ctx.grouped else 'ungrouped'}_{ctx.kmer_prefix}_{ctx.kmer_suffix_size}_{seed}"
	path = f'{ctx.output_directory}/{dataset_name}.csv'
	results.to_csv(path)
	print(f'Saved tmp result to {path=}')
	print(f'{results=}')
	return results



if __name__ == "__main__":

	
	parser = ArgParser(module = "pca_analysis")
	parser = parser.parser
	
	
	phenotypes = parser.phenotype

	if torch.cuda.is_available():
		device = torch.device("cuda")
		
	elif torch.backends.mps.is_available(): 
		device = torch.device("mps")

	else: 
		# On CPU server
		device = torch.device("cpu")

	for phenotype in phenotypes:
		if not parser.extract_feature_importance and not parser.classify and not parser.plot_pca:
			raise ValueError("No action specified - at least one of --classify, --plot_pca or --extract_feature_importance must be set. Aborting...")

		label_return = load_labels(file_path=parser.labels_path, id = parser.id_column, label = phenotype, sep = ",")
		label_dict_literal, label_dict, int2label = label_return["label_dict"], label_return["label_dict_int"], label_return["int2label"] 

		kmer_prefix = parser.kmer_prefix
		kmer_suffix_size = parser.kmer_suffix_size
		
		X, y, groups = embed_data(label_dict=label_dict, 
					input_data_directory=parser.input,
					output_data_directory=parser.output,
					kmer_prefix=parser.kmer_prefix, 
					kmer_suffix_size = parser.kmer_suffix_size, 
					kmer_offset=parser.kmer_offset,
					id_column = parser.id_column,
					sequence_column = parser.dna_sequence_column,
					cores = parser.cores, 
					embedding_class = parser.embedding,
					reembed=parser.reembed,
					file_type=parser.file_type,
					esmc_model=parser.esmc_model,
					esmc_pooling=parser.esmc_pooling,
					device=device,
					group_clusters=parser.group_clusters,
					)
		

		reembed = False  # only reembed once per dataset
		

		ctx = model_context(
							X,
							y, 
							groups,
							parser.group_clusters,
							parser.output,
							phenotype, 
							kmer_prefix, 
							kmer_suffix_size,
							model_type=None,
							int2label=int2label,
							k_folds=parser.k_folds,
							embedding_class=parser.embedding,
							train_split_method=parser.train_split_method)
		
		
		# # Plotting pca and umap
		if parser.plot_pca:
			pca_plot(ctx)
		# # umap_plot(ctx)

		if parser.classify:
			random_forest_classification(ctx)

			hist_gradient_boosting_classifier(ctx)

		if parser.extract_feature_importance:

			feature_importance_extraction(ctx) # Feature extraction


		
		

		

		