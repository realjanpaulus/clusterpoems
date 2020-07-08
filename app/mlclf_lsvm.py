#!/usr/bin/env python
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import logging
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score 
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from stop_words import get_stop_words
import sys
import time


def main():

	# ================
	# time managment #
	# ================

	program_st = time.time()
	clf_durations = defaultdict(list)

	# =======================
	# predefined parameters #
	# =======================

	n_jobs = args.n_jobs
	cv = 5
	cv_dict = {}
	vectorizer = TfidfVectorizer(max_df=0.5,
								 lowercase=args.lowercase,
								 max_features=args.max_features,
								 stop_words=get_stop_words("de"))

	
	# ================================
	# classification logging handler #
	# ================================
	logging_filename = f"../logs/mlclf_lsvm_{args.corpus_name}.log"
	logging.basicConfig(level=logging.DEBUG, filename=logging_filename, filemode="w")
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(levelname)s: %(message)s")
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)

	
	# =================
	# corpus loading  # 
	# =================

	if args.epoch_division == "amann":
		corpus = pd.read_csv("../corpora/amann_poems.csv", index_col=0)
		logging.info(f"Read preload corpus with epoch division by '{args.epoch_division}'.")
	else:
		logging.warning(f"Couldn't find a corpus with the name '{args.corpus_name}'.")
	
	text = corpus["poem"].values
	labels = LabelEncoder().fit_transform(corpus["epoch"].values)
	classes = corpus["epoch"]
	unique_epochs = list(np.unique(corpus["epoch"]))

	logging.info(f"Read {args.corpus_name} corpus ({int((time.time() - program_st)/60)} minute(s)).")
	
	# ================
	# classification # 
	# ================


	# ============
	# Linear SVM #
	# ============

	lsvm_st = time.time()
	lsvm_pipe = Pipeline(steps=[("vect", vectorizer),
								("clf", LinearSVC())])
	
	lsvm_parameters = {"vect__ngram_range": [(1,1)],
					   "clf__penalty": ["l2"],
					   "clf__loss": ["squared_hinge"],
					   "clf__tol": [1e-5, 1e-3],
					   "clf__C": [1.0, 3.0, 5.0],
					   "clf__max_iter": [1000, 3000, 5000],
					   "clf__class_weight": [None, "balanced"]}
	
	#ALTERNATIVE
	"""
	lsvm_parameters = {"clf__penalty": ["l2"],
					   "clf__loss": ["squared_hinge"],
					   "clf__tol": [1e-3],
					   "clf__C": [1.0],
					   "clf__max_iter": [1000]}
	"""

	

	lsvm_grid = GridSearchCV(lsvm_pipe, 
							 lsvm_parameters,
							 cv=cv, 
							 error_score=0.0,
							 n_jobs=args.n_jobs,
							 scoring="f1_macro")

	#lsvm_grid2.fit(features, class2)
	

	lsvm_cv_scores = cross_validate(lsvm_grid, 
									text, 
									labels,
									cv=cv, 
									n_jobs=args.n_jobs,
									return_estimator=False,
									scoring="f1_macro")


	lsvm_preds = cross_val_predict(lsvm_grid, 
								   text, 
								   classes, 
								   cv=cv, 
								   n_jobs=args.n_jobs)
	
	
	conf_mat = confusion_matrix(classes, lsvm_preds)
	cm_df = pd.DataFrame(conf_mat, index=unique_epochs, columns=unique_epochs)


	if args.save_date:
		output_path = f"../results/lsvm_cm_{args.epoch_division}({datetime.now():%d.%m.%y}_{datetime.now():%H:%M}).csv"
	else:
		output_path = f"../results/lsvm_cm_{args.epoch_division}.csv"
	cm_df.to_csv(output_path)


	logging.info(f"LSVM score: {np.mean(lsvm_cv_scores['test_score'])}")
	lsvm_duration = float(time.time() - lsvm_st)
	logging.info(f"Run-time LSVM: {lsvm_duration} seconds")

	

	program_duration = float(time.time() - program_st)
	logging.info(f"Run-time: {int(program_duration)/60} minute(s).")
	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="mlclf_lsvm", description="Classification of LSVM.")
	parser.add_argument("--corpus_name", "-cn", type=str, default="poems", help="Indicates the corpus. Default is 'poems'. Another possible value is 'noise'.")
	parser.add_argument("--epoch_division", "-ed", type=str, default="amann", help="Indicates the epoch division method. Possible values are 'amann', brenner'.")
	parser.add_argument("--lowercase", "-l", type=bool, default=True, help="Indicates if words should be lowercased.")
	parser.add_argument("--max_features", "-mf", type=int, default=10000, help="Indicates the number of most frequent words.")
	parser.add_argument("--n_jobs", "-nj", type=int, default=1, help="Indicates the number of processors used for computation.")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	

	args = parser.parse_args()

	main()
