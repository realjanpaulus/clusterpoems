# Clustering poems

In a series of experiments, it was investigated whether it is possible to identify **literary epochs** of **poems** using **text clustering techniques**. The following techniques were used:
- K-Means
- DBSCAN
- Gaussian Mixture Models
- Hierarchical Clustering
- Recurrent Autoencoder


## Project structure

- **`app`**: Contains Python scripts, Jupyter Notebooks and the json file `epochs.json`.
    - `epochs.json`: Stores multiple different literature epoch divisions.
    - `epochs_addition.json`: Special extension of the epoch division of AMANN.
    - `create_linkage.py`: Script for the creation of a linkage matrix.
    - `mlclf_lsvm.py`: Script for the experiments with LSVM.
    - `pipe.py`: A pipeline for multiple clustering algorithms like K-Means, DBSCAN and Gaussian Mixture Models. Applies selected clustering algorithm to a specific corpus, a specific epoch division method and two selected epochs. For more informations about the possible input arguments run the following command in a terminal: `python pipe.py -h`.
    - `run_pipe.py`: Runs `pipe.py` for every possible combination of epochs. For more informations about the possible input arguments run the following command in a terminal: `python run_pipe.py -h`.
    - `utils.py`: Stores helper functions for clustering experiments.
    - `clustering_whole_corpus.ipynb`: Clustering of the whole corpus and plotting data points and their allocation.
    - `corpus_modification.ipynb`: Modification of the corpus by adding epoch division, merging poets and improving epoch division. Show corpus token count
    distribution.
    - `evaluation.ipynb`: Calculates scores for clustering results, sorts and summarizes them.
    - `hierarchical_clustering.ipynb`: Hierarchical Clustering experiments.
    - `noise_detection.ipynb`: Various experiments to reduce the noise.
- **`corpora`**: Contains text corpora as `csv` files.
- **`presentation`**: Contains the presentation for the clustering experiments.
- **`results`**: Contains the experiment results as images and json files.


## Connection of presentation and repository

The **appendix** of the presentation (see `presentation` folder) contains an assignment of the presented experiments to the scripts in the `app` folder.

## Virtual Environment Setup

Create and activate the environment (the python version and the environment name can vary at will):

```sh
$ python3.9 -m venv .env
$ source .env/bin/activate
```

To install the project's dependencies, activate the virtual environment and simply run (requires [poetry](https://python-poetry.org/)):

```sh
$ poetry install
```

Alternatively, use the following:

```sh
$ pip install -r requirements.txt
```

Deactivate the environment:

```sh
$ deactivate
```


## Notes

`scikit-learn`, `scikit-learn-extra`, `seaborn`, `umap-learn` and `yellowbrick` aren't part of the `pyproject.toml` due to the support of Apple's M1 chip and has to be installed manually:

```sh
pip install scikit-learn scikit-learn-extra seaborn umap-learn yellowbrick
```
