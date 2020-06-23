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
from utils import add_epoch_division, text_cleaning


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
	else:
		logging.warning(f"Couldn't find a corpus with the name '{args.corpus_name}'.")
	

	with open("epochs.json") as f:
		epochs = json.loads(f.read())

	if args.epoch_division == "brenner":
		epochs = epochs["brenner"]
		epoch_exception = args.epoch_exception
		corpus = add_epoch_division(corpus, epochs, epoch_exception=epoch_exception)
		logging.info(f"Added epoch division by '{args.epoch_division}'.")
	else:
		logging.warning(f"Couldn't find a epoch division with the name '{args.epoch_division}'.")


	epoch1 = args.epoch_one
	epoch2 = args.epoch_two
	corpus = corpus[(corpus.epoch == epoch1) | (corpus.epoch == epoch2)]
	logging.info(f"Used epochs are '{epoch1}' and '{epoch2}'.")


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

	kmeans_st = time.time()
	kmeans = KMeans(n_clusters=len(unique_epochs),
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


	output_path = "../results/kmeans_top_words.json"

	if args.reduce_dimensionality:
		output_path = "../results/kmeans_top_words_rd.json"	


	with open(output_path, "r+") as f:
		dic = json.load(f)
		dic[f"{epoch1}/{epoch2}"] = top_words_cluster
		f.seek(0)
		json.dump(dic, f)



	print("--------------- Metrics (K-Means) ---------------")
	kmeans_ars = adjusted_rand_score(labels, kmeans.labels_)
	logging.info(f"Adjusted Rand Score for K-Means: {kmeans_ars}.")

	kmeans_vm = v_measure_score(labels, kmeans.labels_)
	logging.info(f"V-measure for K-Means: {kmeans_vm}.")

	output_path = "../results/kmeans_results.json"

	if args.reduce_dimensionality:
		output_path = "../results/kmeans_results_rd.json"	


	with open(output_path, "r+") as f:
		dic = json.load(f)
		dic[f"{epoch1}/{epoch2}"] = {"ars": kmeans_ars, "vm": kmeans_vm}
		f.seek(0)
		json.dump(dic, f)

	kmeans_amis = adjusted_mutual_info_score(labels, kmeans.labels_)
	logging.info(f"Adjusted Mutuial Info Score for K-Means: {kmeans_amis}.")

	kmeans_hs = homogeneity_score(labels, kmeans.labels_)
	logging.info(f"Homogeneity Score for K-Means: {kmeans_hs}.")

	kmeans_cs = completeness_score(labels, kmeans.labels_)
	logging.info(f"Completeness Score for K-Means: {kmeans_cs}.")
	print("--------------------------------------------------")

	kmeans_duration = float(time.time() - kmeans_st)
	logging.info(f"Run-time K-Means: {kmeans_duration} seconds")

	"""TODO
	dbscan_st = time.time()
	dbscan = DBSCAN(n_jobs=n_jobs)
	dbscan.fit(vector)
	dbscan_ars = adjusted_rand_score(labels, dbscan.labels_)
	logging.info(f"Adjusted Rand Score for DBSCAN: {dbscan_ars}.")
	dbscan_duration = float(time.time() - dbscan_st)
	logging.info(f"Run-time DBSCAN: {dbscan_duration} seconds")
	"""

	"""TODO
	gmm_st = time.time()
	gmm = GaussianMixture()
	gmm.fit(vector)
	gmm_ars = adjusted_rand_score(labels, gmm.labels_)
	logging.info(f"Adjusted Rand Score for Gaussian Mixture Model: {gmm_ars}.")
	gmm_duration = float(time.time() - gmm_st)
	logging.info(f"Run-time Gaussian Mixture Model: {gmm_duration} seconds")
	"""
	

	logging.info(f"Read {args.corpus_name} corpus ({int((time.time() - program_st)/60)} minute(s)).")


	# ============
	# clustering #
	# ============

	
	program_duration = float(time.time() - program_st)
	logging.info(f"Run-time: {int(program_duration)/60} minute(s).")



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="pipe", description="Pipeline for clustering.")
	parser.add_argument("--corpus_name", "-cn", type=str, default="poems", help="Indicates the corpus. Default is 'poems'.")
	parser.add_argument("--epoch_division", "-ed", type=str, default="brenner", help="Indicates the epoch division method.")
	parser.add_argument("--epoch_exception", "-ee", type=str, default="Klassik_Romantik", help="Indicates the epoch which should be skipped.")
	parser.add_argument("--epoch_one", "-eo", type=str, default="Aufklärung", help="Name of the first epoch.")
	parser.add_argument("--epoch_two", "-et", type=str, default="Realismus", help="Name of the first epoch.")
	parser.add_argument("--lowercase", "-l", type=bool, default=True, help="Indicates if words should be lowercased.")
	parser.add_argument("--max_features", "-mf", type=int, default=10000, help="Indicates the number of most frequent words.")
	parser.add_argument("--n_jobs", "-nj", type=int, default=1, help="Indicates the number of processors used for computation.")
	parser.add_argument("--reduce_dimensionality", "-rd", action="store_true", help="Indicates if dimension reduction should be applied before clustering.")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	
	args = parser.parse_args()

	main()
