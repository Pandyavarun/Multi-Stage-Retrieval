from sklearn.metrics import ndcg_score
import numpy as np

def calculate_ndcg(relevance_scores, predicted_ranks, k=10):
    true_relevance = np.asarray([relevance_scores])
    predicted_relevance = np.asarray([predicted_ranks])
    return ndcg_score(true_relevance, predicted_relevance, k=k)
