# clustering
Clustering files for the course "Clustering" at the Julius-Maximilians University Würzburg, SS20.


## Project structure

**app**: Contains Python scripts, Jupyter Notebooks and the json file `epochs.json`.
- `epochs.json`: Stores multiple different literature epoch divisions.
- `pipe.py`: A pipeline for multiple clustering algorithms like k-Means, DBSCAN and Gaussian Mixture Models. Applies selected clustering algorithm to a specific corpus, a specific epoch division method and two selected epochs. For more informations about the possible input arguments run the following command in a terminal: `python pipe.py -h`. 
- `run_pipe.py`: Runs `pipe.py` for every possible combination of epochs. For more informations about the possible input arguments run the following command in a terminal: `python run_pipe.py -h`. 
- `utils.py`: Stores helper functions for clustering experiments.


