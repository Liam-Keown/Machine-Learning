import torch
import pickle
import numpy as np
from typing import List, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN


def load_with_pickle(filename: str) -> Optional[Tuple[List[str], str]]:
    """
    Loads a Python object from a file using the pickle module.
    Handles a list of strings or a list of lists of strings.

    Args:
        filename (str): The name of the file to load from.

    Returns:
        Optional[Tuple[List[str], str]]: The loaded data tuple, or None if an error occurred.
    """
    print(f"\nLoading data from '{filename}' using pickle...")
    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        print("Data loaded successfully.")

    except:
        FileNotFoundError

    return data

def perform_clustering_dbscan(reviews_with_labels: List[Tuple[List[str], str]], eps, min_samples):
    """
    Performs DBSCAN clustering on a list of text reviews.
    
    Args:
        reviews_with_labels (List[Tuple[List[str], str]]): A list where each item is a tuple.
                                                          The first element is a list of pre-processed review words,
                                                          and the second is a label string.
        eps (float): The maximum distance between two samples for one to be
                     considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point
                           to be considered as a core point.
    """
    if not reviews_with_labels:
        print("No reviews found for clustering.")
        return

    reviews = [" ".join(review_words) for review_words, label in reviews_with_labels]

    print("\n--- Performing DBSCAN Clustering ---")
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(reviews)
    
    print(f"Reviews vectorized. Data shape: {X.shape}")
    
    # Run DBSCAN
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
    labels = model.fit_predict(X)
    
    # Print results
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"Clustering completed.")
    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Estimated number of noise points: {n_noise}")
    
    if n_clusters > 0:
        print("\n--- Top terms for each cluster ---")
        cluster_indices = [np.where(labels == i)[0] for i in unique_labels if i != -1]
        terms = vectorizer.get_feature_names_out()

        for i, indices in enumerate(cluster_indices):

            centroid = X[indices].mean(axis=0).A1

            top_term_indices = np.argsort(centroid)[::-1][:10]
            top_terms = [terms[ind] for ind in top_term_indices]
            
            print(f"Cluster {i} top terms:")
            print(f"  {', '.join(top_terms)}")

# --- Main ---
if __name__ == "__main__":
    reviews_data_tuple = load_with_pickle("processed_unsup_reviews.pkl")

    if reviews_data_tuple:
        perform_clustering_dbscan(reviews_data_tuple, eps=0.75, min_samples=3)
