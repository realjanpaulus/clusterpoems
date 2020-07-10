import json
import matplotlib.pyplot as plt
from nltk import word_tokenize
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from stop_words import get_stop_words
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

from utils import add_epoch_division, alter_epoch_division, linkage_matrix, plot_dendrogram, remove_noise_poet, text_cleaning

DIM_RED = False
LOWERCASE = True
MAX_FEATURES = 10000
PATH = "../corpora/amann_poems.csv"
REDUCE_CORPUS = False
STOP_WORDS = get_stop_words("de")


orig_corpus = text_cleaning(pd.read_csv("../corpora/german_poems.csv"))

with open("epochs.json") as f:
    epochs = json.loads(f.read())
    
with open("epochs_addition.json") as f:
    alternative_epochs = json.loads(f.read())
    
epochs = epochs["amann"]
epoch_exceptions = ["Sturm_Drang"]
orig_corpus = add_epoch_division(orig_corpus, epochs, epoch_exceptions=epoch_exceptions)
orig_corpus = alter_epoch_division(orig_corpus, alternative_epochs)


vectorizer = TfidfVectorizer(max_df=0.5,
                             lowercase=LOWERCASE,
                             max_features=MAX_FEATURES,
                             stop_words=STOP_WORDS)
orig_vector = vectorizer.fit_transform(orig_corpus["poem"])


orig_text = orig_corpus["poem"]
orig_shortened_classes = [c[:2] for c in orig_corpus["epoch"].values]
orig_pids = [p for p in orig_corpus["pid"].values]


if DIM_RED:
    pca = PCA(n_components=3)
    orig_X_red = pca.fit_transform(orig_vector.toarray())
else:
    orig_X_red = orig_vector.toarray()

print("Starting Clustering.")
orig_agcl = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
orig_model = orig_agcl.fit(orig_X_red)


linkage = pd.DataFrame(linkage_matrix(model))

linkage.to_csv("../results/linkage_matrix_poems.csv", index=False)

print("Finished.")



