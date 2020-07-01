import argparse
from itertools import combinations
import json
import logging
import subprocess
import sys
import time

from utils import clear_json

def main():

	program_st = time.time()


	logging.basicConfig(level=logging.DEBUG, 
						filename=f"../logs/run_{args.corpus_name}.log", 
						filemode="w")
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(levelname)s: %(message)s")
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)
	

	logging.info("Clustering for all epoch combinations.")


	# reset json files #

	output_path = "../results/kmeans_top_words.json"
	if args.reduce_dimensionality:
		output_path = "../results/kmeans_top_words_rd.json"	

	if args.clear_json:
		clear_json(output_path)


	output_path = "../results/kmeans_results.json"
	if args.reduce_dimensionality:
		output_path = "../results/kmeans_results_rd.json"	

	if args.clear_json:
		clear_json(output_path)

	if args.epoch_division == "brenner":
		epochs = {
					"Barock": {"b": 1600, "e": 1700},
					"Frühaufklärung": {"b": 1700, "e": 1755},
					"Aufklärung": {"b": 1755, "e": 1810},
					"Klassik_Romantik": {"b": 1786, "e": 1832},
					"Biedermeier": {"b": 1815, "e": 1848},
					"Realismus": {"b": 1848, "e": 1900},
					"Moderne": {"b": 1880, "e": 1918},
					"Weimarer_Republik": {"b": 1918, "e": 1933}
				}

		unique_epochs = list(epochs.keys())

		if args.epoch_exception in unique_epochs:
			unique_epochs.remove(args.epoch_exception)



	combinations_inputs = list(combinations(unique_epochs, r=2))


	for idx, t in enumerate(combinations_inputs):
		
		print("--------------------------------------------")
		logging.info(f"Argument combination {idx+1}/{len(combinations_inputs)}.")
		logging.info(f"Epoch 1: {t[0]}.")
		logging.info(f"Epoch 2: {t[1]}.")
		print("--------------------------------------------")


		command = f"python pipe.py -cn {args.corpus_name} -ed {args.epoch_division} -ee {args.epoch_exception} -eo {t[0]} -et {t[1]} -l {args.lowercase} -m {args.method} -mf {args.max_features} -nj {args.n_jobs}"

		
		if args.save_date:
			command += " -sd"

		if args.reduce_dimensionality:
			command += " -rd"

		if args.merge_poet:
			command += " -mp"

		if args.clear_json:
			command += " -cj"


		subprocess.call(["bash", "-c", command])
		print("\n")
	program_duration = float(time.time() - program_st)
	logging.info(f"Overall run-time: {int(program_duration)/60} minute(s).")

	
	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="run_pipe", description="Runs clustering script with multiple epoch combinations.")
	parser.add_argument("--clear_json", "-cj", action="store_true", help="Indicates if previous json results should cleared.")
	parser.add_argument("--corpus_name", "-cn", type=str, default="poems", help="Indicates the corpus. Default is 'poems'.")
	parser.add_argument("--epoch_division", "-ed", type=str, default="brenner", help="Indicates the epoch division method.")
	parser.add_argument("--epoch_exception", "-ee", type=str, default="Klassik_Romantik", help="Indicates the epoch which should be skipped.")
	parser.add_argument("--lowercase", "-l", type=bool, default=True, help="Indicates if words should be lowercased.")
	parser.add_argument("--max_features", "-mf", type=int, default=10000, help="Indicates the number of most frequent words.")
	parser.add_argument("--merge_poet", "-mp", action="store_true", help="Indicates if all poems of a poet should be merged.")
	parser.add_argument("--method", "-m", type=str, default="kmeans", help="Indicates clustering method. Possible values are 'kmeans', 'dbscan', 'gmm', 'all'.")
	parser.add_argument("--n_jobs", "-nj", type=int, default=1, help="Indicates the number of processors used for computation.")
	parser.add_argument("--reduce_dimensionality", "-rd", action="store_true", help="Indicates if dimension reduction should be applied before clustering.")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	
	args = parser.parse_args()

	main()