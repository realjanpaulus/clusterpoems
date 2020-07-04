# clustering
Clustering files for the course "Clustering" at the Julius-Maximilians University Würzburg, SS20.


## Project structure

**app**: Contains Python scripts, Jupyter Notebooks and the json file `epochs.json`.
- `epochs.json`: Stores multiple different literature epoch divisions.
- `pipe.py`: A pipeline for multiple clustering algorithms like k-Means, DBSCAN and Gaussian Mixture Models. Applies selected clustering algorithm to a specific corpus, a specific epoch division method and two selected epochs. For more informations about the possible input arguments run the following command in a terminal: `python pipe.py -h`. 
- `run_pipe.py`: Runs `pipe.py` for every possible combination of epochs. For more informations about the possible input arguments run the following command in a terminal: `python run_pipe.py -h`. 
- `utils.py`: Stores helper functions for clustering experiments.

**corpora**: Contains text corpora in csv files.

**presentation**: Contains the presentation for the clustering experiments.

**results**: Contains the experiment results as images and json files.

## TODOs:

- [x] Text cleaning erweitert
- [x] Andere Art der Zusammenfassung von Gedichten: Dichter mit mehreren Epochenzuweisungen nach diesen aufteilen
- Noise entdecken durch Hierarchisches Clustering inkl. Epochenzuteilungen
- Korpus normalisieren und Experimente durchführen
- Analyse von Gedichten/Dichter, die sich gut clustern lassen
- Topic Modelling anwenden: erzeugen epochen bestimmte topics?
- Topic Modelling anwenden als downstream task für klassifizierung?
- DBSCAN anwenden
- mit veränderter zusammenfassung mal clustering auf gesamten korpus
- andere zuteilung finden anstatt brenner
- guideline schreiben für stelle in repo, die für gute ergebnisse verantwortlich waren
- schreiben, dass clustering von korpus so nicht funktioniert hat (mit erklärung)
- pca und umap experimente (kein truncated svd)
- mit pca bisschen rumspielen, um einblick in daten zu bekommen
- kmeans erweitern mit cosinus distance?












