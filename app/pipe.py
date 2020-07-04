#!/usr/bin/env python
"""TODO
- mehr corpora
- mehr Epocheneinteilungen
- dimensionality reduction
- KernelPCA zur Visualisierung nutzen? 
	- siehe GERON, dimension reduction, 226f.
	- da dann acuh GridSearch machen (GERON, 228)
- Durchgehen: https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
	- vllt noch weitere, interessante Evaluationsmetriken
	- nochmal klar machen, was welche metrik verwendet
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import logging
import numpy as np
import pandas as pd


from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score, v_measure_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer


from stop_words import get_stop_words
import sys
import time
from utils import add_epoch_division, clear_json, merge_corpus_poets, text_cleaning
from yellowbrick.text import UMAPVisualizer

def main():

	# ================
	# time managment #
	# ================

	program_st = time.time()

	# =======================
	# predefined parameters #
	# =======================

	n_jobs = args.n_jobs
	n_components = 3 #for dimension reduction
	results_dict = {}
	top_words = {}

	
	# ================================
	# classification logging handler #
	# ================================
	logging_filename = f"../logs/pipe_{args.corpus_name}.log"
	logging.basicConfig(level=logging.DEBUG, filename=logging_filename, filemode="w")
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(levelname)s: %(message)s")
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)

	
	# =================
	# corpus loading  # 
	# =================

	if args.corpus_name == "poems":
		corpus = text_cleaning(pd.read_csv("../corpora/german_poems.csv"))
		text_name = "poem"
		logging.info(f"Read '{args.corpus_name}' corpus.")
	elif args.corpus_name == "noise":
		corpus = pd.read_csv("../corpora/german_poems_noiseless.csv")
		text_name = "poem"
		logging.info(f"Read '{args.corpus_name}' corpus.")
	else:
		logging.warning(f"Couldn't find a corpus with the name '{args.corpus_name}'.")
	

	with open("epochs.json") as f:
		epochs = json.loads(f.read())

	#TODO: mehr
	if args.epoch_division == "amann":
		epochs = epochs["amann"]
		epoch_exceptions = ["Sturm_Drang"]
		corpus = add_epoch_division(corpus, epochs, epoch_exceptions=epoch_exceptions)
		logging.info(f"Added epoch division by '{args.epoch_division}'.")
	elif args.epoch_division == "brenner":
		epochs = epochs["brenner"]
		epoch_exceptions = ["Klassik_Romantik"]
		corpus = add_epoch_division(corpus, epochs, epoch_exceptions=epoch_exceptions)
		logging.info(f"Added epoch division by '{args.epoch_division}'.")
	elif args.epoch_division == "simple":
		epochs = epochs["simple"]
		corpus = add_epoch_division(corpus, epochs, epoch_exceptions="")
		logging.info(f"Added epoch division by '{args.epoch_division}'.")
	else:
		logging.warning(f"Couldn't find a epoch division with the name '{args.epoch_division}'.")



	if args.merge_poet:
		corpus = merge_corpus_poets(corpus)

	epoch1 = args.epoch_one
	epoch2 = args.epoch_two
	corpus = corpus[(corpus.epoch == epoch1) | (corpus.epoch == epoch2)]
	logging.info(f"Used epochs are '{epoch1}' and '{epoch2}'.")
	logging.info(f"Count of different poets: {len(np.unique(corpus.poet))}")
	logging.info(f"Count of poets of epoch '{epoch1}': {len(np.unique(corpus[corpus.epoch == epoch1].poet))}")
	logging.info(f"Count of poets of epoch '{epoch2}': {len(np.unique(corpus[corpus.epoch == epoch2].poet))}")

	text = corpus[text_name].values
	labels = LabelEncoder().fit_transform(corpus["epoch"].values)
	unique_epochs = list(np.unique(corpus["epoch"]))


	
	# ===============
	# vectorization #
	# ===============

	#TODO: min, max df to args?!
	vectorizer = TfidfVectorizer(max_df=0.5,
								 lowercase=args.lowercase,
								 max_features=args.max_features,
								 stop_words=get_stop_words("de"))
	vector = vectorizer.fit_transform(text)

	if args.reduce_dimensionality:
		logging.info(f"Reduce dimensionality to {n_components}.")
		svd = TruncatedSVD(n_components=n_components)
		normalizer = Normalizer(copy=False)
		lsa = make_pipeline(svd, normalizer)
		vector = lsa.fit_transform(vector)

		explained_variance = svd.explained_variance_ratio_.sum()
		logging.info("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))


	# ============
	# clustering #
	# ============


	if args.method == "kmeans" or args.method == "all":

		kmeans_st = time.time()
		kmeans = KMeans(len(unique_epochs),
						n_jobs=n_jobs)
		kmeans.fit(vector)


		# ==================
		# kmeans top words #
		# ==================

		if args.reduce_dimensionality:
			original_space_centroids = svd.inverse_transform(kmeans.cluster_centers_)
			order_centroids = original_space_centroids.argsort()[:, ::-1]
		else:
			order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

		terms = vectorizer.get_feature_names()

		top_words_cluster = {}
		for i in range(len(unique_epochs)):
			top_words_cluster[i] = [terms[ind] for ind in order_centroids[i, :10]]


	
		print("--------------- Metrics (K-Means) ---------------")
		kmeans_ari = adjusted_rand_score(labels, kmeans.labels_)
		logging.info(f"Adjusted Rand Score for K-Means: {kmeans_ari}.")

		kmeans_vm = v_measure_score(labels, kmeans.labels_)
		logging.info(f"V-measure for K-Means: {kmeans_vm}.")
		print("--------------------------------------------------")


		output_name = f"kmeans_results_{args.epoch_divison}"

		if args.reduce_dimensionality:
			output_name += "_rd"

		if args.save_date:
			output_name += f"({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"

		output_path = f"../results/{output_name}.json"

		
		with open(output_path, "w+") as f:
			try:
				dic = json.load(f)
			except:
				dic = {}
			dic[f"{epoch1}/{epoch2}"] = {"scores": {"ari": kmeans_ari, 
												 	"vm": kmeans_vm},
									 	 "tw": top_words_cluster}
			f.seek(0)
			json.dump(dic, f)


		if args.clear_json:
			clear_json(output_path)

		
		kmeans_duration = float(time.time() - kmeans_st)
		logging.info(f"Run-time K-Means: {kmeans_duration} seconds")
	elif args.method == "dbscan" or args.method == "all":

	
		dbscan_st = time.time()
		dbscan = DBSCAN(n_jobs=n_jobs)
		dbscan.fit(vector)
		dbscan_ari = adjusted_rand_score(labels, dbscan.labels_)
		logging.info(f"Adjusted Rand Score for DBSCAN: {dbscan_ari}.")
		dbscan_duration = float(time.time() - dbscan_st)
		logging.info(f"Run-time DBSCAN: {dbscan_duration} seconds")
	elif args.method == "gmm" or args.method == "all":

		gmm_st = time.time()
		gmm = GaussianMixture(n_components=2, n_init=10, max_iter=100)
		gmm.fit(vector.toarray())

		if not gmm.converged_:
			logging.info("Gaussian Mixture Model didn't converged. Increase max iter.")
			gmm = GaussianMixture(n_components=2, n_init=10, max_iter=250)
			gmm.fit(vector.toarray())

		gmm_labels = gmm.predict(vector.toarray())

		# ==================
		# gmm top words #
		# ==================

		# TODO


		print("--------------- Metrics (Gaussian Mixture Model) ---------------")
		gmm_ari = adjusted_rand_score(labels, gmm_labels)
		logging.info(f"Adjusted Rand Score for Gaussian Mixture Model: {gmm_ari}.")
		

		gmm_vm = v_measure_score(labels, gmm_labels)
		logging.info(f"V-measure for for Gaussian Mixture Model: {gmm_vm}.")

		output_name = "gmm_results"

		if args.reduce_dimensionality:
			output_name += "_rd"

		if args.save_date:
			output_name += f"({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"

		
		output_path = f"../results/{output_name}.json"

		if args.keep_json:
			clear_json(output_path)

		with open(output_path, "r+") as f:
			dic = json.load(f)
			dic[f"{epoch1}/{epoch2}"] = {"ari": gmm_ari, "vm": gmm_vm}
			f.seek(0)
			json.dump(dic, f)

		print("-----------------------------------------------------------------")

		gmm_duration = float(time.time() - gmm_st)
		logging.info(f"Run-time Gaussian Mixture Model: {gmm_duration} seconds")
	else:
		logging.warning(f"Couldn't find a method with the name '{args.method}'.")

	
	if args.visualization:
		umap = UMAPVisualizer(color="bold")
		umap.fit(vector, list(corpus.epoch))

		figure_name = f"{epoch1}_{epoch2}"

		if args.reduce_dimensionality:
			figure_name += "_rd"

		if args.save_date:
			figure_name += f"({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"


		figure_path = f"../results/figures/{figure_name}.png"
		umap.show(outpath=figure_path, dpi=300)


	
	program_duration = float(time.time() - program_st)
	logging.info(f"Run-time: {int(program_duration)/60} minute(s).")



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="pipe", description="Pipeline for clustering.", add_help=True)
	parser.add_argument("--clear_json", "-cj", action="store_true", help="Indicates if previous json results should cleared.")
	parser.add_argument("--corpus_name", "-cn", type=str, default="poems", help="Indicates the corpus. Default is 'poems'. Another possible value is 'noise'.")
	parser.add_argument("--epoch_division", "-ed", type=str, default="brenner", help="Indicates the epoch division method. Possible values are 'amann', brenner'.")
	#parser.add_argument("--epoch_exception", "-ee", type=str, default="Klassik_Romantik", help="Indicates the epoch which should be skipped.")
	parser.add_argument("--epoch_one", "-eo", type=str, default="Barock", help="Name of the first epoch.")
	parser.add_argument("--epoch_two", "-et", type=str, default="Realismus", help="Name of the first epoch.")
	parser.add_argument("--lowercase", "-l", type=bool, default=True, help="Indicates if words should be lowercased.")
	parser.add_argument("--max_features", "-mf", type=int, default=10000, help="Indicates the number of most frequent words.")
	parser.add_argument("--merge_poet", "-mp", action="store_true", help="Indicates if all poems of a poet should be merged.")
	parser.add_argument("--method", "-m", type=str, default="kmeans", help="Indicates clustering method. Possible values are 'kmeans', 'dbscan', 'gmm', 'all'.")
	parser.add_argument("--n_jobs", "-nj", type=int, default=1, help="Indicates the number of processors used for computation.")
	parser.add_argument("--reduce_dimensionality", "-rd", action="store_true", help="Indicates if dimension reduction should be applied before clustering.")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	parser.add_argument("--visualization", "-v", action="store_true", help="Indicates if results should be visualized.")

	args = parser.parse_args()

	main()
