import numpy as np


def cosine_similarity(s, g):
    similarity = np.sum(s * g, axis=1) / np.sqrt((np.sum(s * s, axis=1) * np.sum(g * g, axis=1)))

    # return np.sum(similarity)
    return similarity


def novel_metrics(samples, ground_truth):

    # samples, ground_truth: [n_samples]
    return cosine_similarity(np.array(samples), np.array(ground_truth))
