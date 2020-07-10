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
from itertools import product
import json
import logging
from nltk import word_tokenize
import numpy as np
import os
import pandas as pd


from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, v_measure_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer

from sklearn_extra.cluster import KMedoids

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
	n_components = 3 # for dimensionality reduction
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

	if args.epoch_division == "amann":
		if args.preload:
			corpus = pd.read_csv("../corpora/amann_poems.csv", index_col=0)
			logging.info(f"Read preload corpus with epoch division by '{args.epoch_division}'.")
		else:
			epochs = epochs["amann"]
			epoch_exceptions = ["Sturm_Drang"]
			corpus = add_epoch_division(corpus, epochs, epoch_exceptions=epoch_exceptions)
			logging.info(f"Added epoch division by '{args.epoch_division}'.")

			if args.merge_poet:
				corpus = merge_corpus_poets(corpus)
				corpus["poemlength"] = corpus.poem.apply(lambda x: len(word_tokenize(x)))
				corpus = corpus[corpus.poemlength >= 1000]
	elif args.epoch_division == "amann_noise":
		corpus = pd.read_csv("../corpora/amann_poems_noiseless.csv", index_col=0)
		logging.info(f"Read preload corpus with epoch division by '{args.epoch_division}'.")
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


		output_name = f"kmeans_results_{args.epoch_division}"

		if args.reduce_dimensionality:
			output_name += "_rd"

		if args.save_date:
			output_name += f"({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"

		output_path = f"../results/{output_name}.json"

		
		if os.path.exists(output_path):
			logging.info("Update results file.")
			with open(output_path, "r+") as f:
				dic = json.load(f)
				dic[f"{epoch1}/{epoch2}"] = {"scores": {"ari": kmeans_ari, 
													"vm": kmeans_vm},
											 "tw": top_words_cluster}
				f.seek(0)
				f.write(json.dumps(dic))
				f.truncate()
		else:
			logging.info("Create results file.")
			with open(output_path, "w") as f:
				dic = {}
				dic[f"{epoch1}/{epoch2}"] = {"scores": {"ari": kmeans_ari, 
													"vm": kmeans_vm},
											 "tw": top_words_cluster}
				json.dump(dic, f)
		

		if args.clear_json:
			clear_json(output_path)

		
		kmeans_duration = float(time.time() - kmeans_st)
		logging.info(f"Run-time K-Means: {kmeans_duration} seconds")
	
	if args.method == "kmedoids" or args.method == "all":
		kmedoids_st = time.time()
		kmedoids = KMedoids(n_clusters=len(unique_epochs),
							init="k-medoids++",
							metric="cosine")
		kmedoids.fit(vector)



	
		print("--------------- Metrics (K-Medoids) ---------------")
		kmedoids_ari = adjusted_rand_score(labels, kmedoids.labels_)
		logging.info(f"Adjusted Rand Score for K-Medoids: {kmedoids_ari}.")

		kmedoids_vm = v_measure_score(labels, kmedoids.labels_)
		logging.info(f"V-measure for K-Medoids: {kmedoids_vm}.")
		print("--------------------------------------------------")


		output_name = f"kmedoids_results_{args.epoch_division}"

		if args.reduce_dimensionality:
			output_name += "_rd"

		if args.save_date:
			output_name += f"({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"

		output_path = f"../results/{output_name}.json"

		
		if os.path.exists(output_path):
			logging.info("Update results file.")
			with open(output_path, "r+") as f:
				dic = json.load(f)
				dic[f"{epoch1}/{epoch2}"] = {"scores": {"ari": kmedoids_ari, 
														"vm": kmedoids_vm}}
				f.seek(0)
				f.write(json.dumps(dic))
				f.truncate()
		else:
			logging.info("Create results file.")
			with open(output_path, "w") as f:
				dic = {}
				dic[f"{epoch1}/{epoch2}"] = {"scores": {"ari": kmedoids_ari, 
														"vm": kmedoids_vm}}
				json.dump(dic, f)
		

		if args.clear_json:
			clear_json(output_path)

		
		kmedoids_duration = float(time.time() - kmedoids_st)
		logging.info(f"Run-time K-Medoids: {kmedoids_duration} seconds")
	
	if args.method == "dbscan" or args.method == "all":
		dbscan_st = time.time()

		if args.use_tuning:
			logging.info("Tuning hyperparameters for dbscan.")
			eps_search = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
			min_samples = [2, 3, 4, 5, 6, 7]
			metrics = ["cosine", "euclidean"]

			dbscan_best_params = {"ari": 0,
								  "vm": 0,
								  "params": ()}

			cartesian_inputs = list(product(eps_search, min_samples, metrics))

			for t in cartesian_inputs:
				dbscan = DBSCAN(eps=t[0],
								min_samples=t[1],
								metric=t[2],
								n_jobs=n_jobs)
				dbscan.fit(vector)
				dbscan_ari = adjusted_rand_score(labels, dbscan.labels_)
				dbscan_vm = v_measure_score(labels, dbscan.labels_)

				prev_ari = dbscan_best_params["ari"]

				if dbscan_ari > prev_ari:
					dbscan_best_params["ari"] = dbscan_ari
					dbscan_best_params["vm"] = dbscan_vm
					dbscan_best_params["params"] = t


			t = dbscan_best_params["params"]
			logging.info(f"Best params for DBSCAN:\neps: {t[0]}\nmin_samples: {t[1]}\nmetrics: {t[2]}")
			dbscan = DBSCAN(eps=t[0],
							min_samples=t[1],
							metric=t[2],
							n_jobs=n_jobs)
			dbscan.fit(vector)

		else:
			dbscan = DBSCAN(n_jobs=n_jobs)
			dbscan.fit(vector)


		print("--------------- Metrics (DBSCAN) ---------------")
		dbscan_ari = adjusted_rand_score(labels, dbscan.labels_)
		logging.info(f"Adjusted Rand Score for DBSCAN: {dbscan_ari}.")
		dbscan_vm = v_measure_score(labels, dbscan.labels_)
		logging.info(f"V-measure for DBSCAN: {dbscan_vm}.")
		print("------------------------------------------------")


		# TODO: top words?


		output_name = f"dbscan_results_{args.epoch_division}"

		if args.reduce_dimensionality:
			output_name += "_rd"

		if args.save_date:
			output_name += f"({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"

		output_path = f"../results/{output_name}.json"

		
		if os.path.exists(output_path):
			logging.info("Update results file.")
			with open(output_path, "r+") as f:
				dic = json.load(f)
				dic[f"{epoch1}/{epoch2}"] = {"scores": {"ari": dbscan_ari, 
														"vm": dbscan_vm}}
				f.seek(0)
				f.write(json.dumps(dic))
				f.truncate()
		else:
			logging.info("Create results file.")
			with open(output_path, "w") as f:
				dic = {}
				dic[f"{epoch1}/{epoch2}"] = {"scores": {"ari": dbscan_ari, 
														"vm": dbscan_vm}}
				json.dump(dic, f)

		dbscan_duration = float(time.time() - dbscan_st)
		logging.info(f"Run-time DBSCAN: {dbscan_duration} seconds")

	if args.method == "gmm":

		gmm_st = time.time()

		if args.use_tuning:
			logging.info("Tuning hyperparameters for gmm.")
			covariance_types = ["full", "tied", "diag", "spherical"]

			gmm_best_params = {"ari": 0,
							   "vm": 0,
							   "params": ()}

			cartesian_inputs = list(product(covariance_types))

			for t in cartesian_inputs:

				gmm = GaussianMixture(n_components=len(unique_epochs), 
								  	  n_init=10,
								  	  covariance_type=t[0], 
								  	  max_iter=100)
				gmm.fit(vector.toarray())

				if not gmm.converged_:
					logging.info("Gaussian Mixture Model didn't converged. Increase max iter.")
					gmm = GaussianMixture(n_components=len(unique_epochs), 
										  n_init=10, 
								  	  	  covariance_type=t[0],
										  max_iter=250)
					gmm.fit(vector.toarray())

				gmm_labels = gmm.predict(vector.toarray())
				gmm_ari = adjusted_rand_score(labels, gmm_labels)
				gmm_vm = v_measure_score(labels, gmm_labels)

				prev_ari = gmm_best_params["ari"]

				if gmm_ari > prev_ari:
					gmm_best_params["ari"] = gmm_ari
					gmm_best_params["vm"] = gmm_vm
					gmm_best_params["params"] = t


			t = gmm_best_params["params"]
			logging.info(f"Best params for GMM:\ncovariance type: {t[0]}")
			gmm = GaussianMixture(n_components=len(unique_epochs), 
								  n_init=10, 
								  covariance_type=t[0],
								  max_iter=250)
			gmm.fit(vector)

		else:
			gmm = GaussianMixture(n_components=len(unique_epochs), 
								  n_init=10, 
								  max_iter=100)
			gmm.fit(vector.toarray())

			if not gmm.converged_:
				logging.info("Gaussian Mixture Model didn't converged. Increase max iter.")
				gmm = GaussianMixture(n_components=len(unique_epochs), 
									  n_init=10, 
									  max_iter=250)
				gmm.fit(vector.toarray())

		gmm_labels = gmm.predict(vector.toarray())

		
		print("--------------- Metrics (Gaussian Mixture Model) ---------------")
		gmm_ari = adjusted_rand_score(labels, gmm_labels)
		logging.info(f"Adjusted Rand Score for Gaussian Mixture Model: {gmm_ari}.")
		

		gmm_vm = v_measure_score(labels, gmm_labels)
		logging.info(f"V-measure for for Gaussian Mixture Model: {gmm_vm}.")
		print("-----------------------------------------------------------------")


		output_name = f"gmm_results_{args.epoch_division}"

		if args.reduce_dimensionality:
			output_name += "_rd"

		if args.save_date:
			output_name += f"({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"

		output_path = f"../results/{output_name}.json"

		if args.clear_json:
			clear_json(output_path)

		if os.path.exists(output_path):
			logging.info("Update results file.")
			with open(output_path, "r+") as f:
				dic = json.load(f)
				dic[f"{epoch1}/{epoch2}"] = {"scores": {"ari": gmm_ari, 
														"vm": gmm_vm}}
				f.seek(0)
				f.write(json.dumps(dic))
				f.truncate()
		else:
			logging.info("Create results file.")
			with open(output_path, "w") as f:
				dic = {}
				dic[f"{epoch1}/{epoch2}"] = {"scores": {"ari": gmm_ari, 
														"vm": gmm_vm}}

		
		gmm_duration = float(time.time() - gmm_st)
		logging.info(f"Run-time Gaussian Mixture Model: {gmm_duration} seconds")
	
	if args.method not in ["all", "kmeans", "kmedoids", "dbscan", "gmm"]:
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
	parser.add_argument("--epoch_division", "-ed", type=str, default="amann", help="Indicates the epoch division method. Possible values are 'amann', brenner'.")
	#parser.add_argument("--epoch_exception", "-ee", type=str, default="Klassik_Romantik", help="Indicates the epoch which should be skipped.")
	parser.add_argument("--epoch_one", "-eo", type=str, default="Barock", help="Name of the first epoch.")
	parser.add_argument("--epoch_two", "-et", type=str, default="Realismus", help="Name of the first epoch.")
	parser.add_argument("--lowercase", "-l", type=bool, default=True, help="Indicates if words should be lowercased.")
	parser.add_argument("--max_features", "-mf", type=int, default=10000, help="Indicates the number of most frequent words.")
	parser.add_argument("--merge_poet", "-mp", action="store_true", help="Indicates if all poems of a poet should be merged.")
	parser.add_argument("--method", "-m", type=str, default="kmeans", help="Indicates clustering method. Possible values are 'kmeans', 'dbscan', 'gmm', 'all'.")
	parser.add_argument("--n_jobs", "-nj", type=int, default=1, help="Indicates the number of processors used for computation.")
	parser.add_argument("--preload", "-p", type=bool, default=True, help="Indicates if preload epoch division should be used.")
	parser.add_argument("--reduce_dimensionality", "-rd", action="store_true", help="Indicates if dimension reduction should be applied before clustering.")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	parser.add_argument("--use_tuning", "-ut", action="store_true", help="Indicates if parameter tuning should be used.")
	parser.add_argument("--visualization", "-v", action="store_true", help="Indicates if results should be visualized.")

	args = parser.parse_args()

	main()
