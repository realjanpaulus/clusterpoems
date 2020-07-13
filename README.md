# clustering
Clustering files for the course "Clustering" at the Julius-Maximilians University Würzburg, SS20.


## Project structure

**app**: Contains Python scripts, Jupyter Notebooks and the json file `epochs.json`.
- `epochs.json`: Stores multiple different literature epoch divisions.

- `pipe.py`: A pipeline for multiple clustering algorithms like k-Means, DBSCAN and Gaussian Mixture Models. Applies selected clustering algorithm to a specific corpus, a specific epoch division method and two selected epochs. For more informations about the possible input arguments run the following command in a terminal: `python pipe.py -h`. 
- `run_pipe.py`: Runs `pipe.py` for every possible combination of epochs. For more informations about the possible input arguments run the following command in a terminal: `python run_pipe.py -h`. 
- `utils.py`: Stores helper functions for clustering experiments.

- `corpus_modification.ipynb`: Modification of the corpus by adding epoch division, merging poets and improving epoch division. Show corpus token count 
distribution. 


**corpora**: Contains text corpora in csv files.

**presentation**: Contains the presentation for the clustering experiments.

**results**: Contains the experiment results as images and json files.

## TODOs:

- [x] Text cleaning erweitert
- [x] Andere Art der Zusammenfassung von Gedichten: Dichter mit mehreren Epochenzuweisungen nach diesen aufteilen
- [x] mit veränderter zusammenfassung mal clustering auf gesamten korpus
- [x] DBSCAN anwenden
- [x] andere zuteilung finden anstatt brenner
- [x] kmeans erweitern mit cosinus distance? --> k-medoids
- [x]Korpus normalisieren und Experimente durchführen
- [x, x] GMM anwenden als Vergleich, vllt nur auf besten Epochenzuweisung!
	- [x] angewandt --> langsam und weniger gute ergebnisse
- [x] **Noise** entdecken
	- [x] Hierarchisches Clustering inkl. Epochenzuteilungen
		- originales Korpus
		- reduziertes Korpus
	- [x] Topic Modelling --> Dialekt
	- [x] LSVM
- [x] POS-TAGGING


- Analyse von Gedichten/Dichter, die sich gut clustern lassen
- Topic Modelling anwenden: erzeugen epochen bestimmte topics?
	- Topics der besten Epochen-Unterscheidungen angucken
	- FRAGE: hier wirklich andere Themen oder nur Rechtschreibung?
- autoencoder
	- dim red: https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/
	- https://github.com/erickrf/autoencoder/tree/master/src
- zeitlichen aspekt untersuchen
- schreiben, dass clustering von korpus so nicht funktioniert hat (mit erklärung)
- evaluation aufräumen
- In Powerpoint Verweise auf Notebooks, Python files etc. im Github-Repo
	- guideline schreiben für stelle in repo, die für gute ergebnisse verantwortlich waren














