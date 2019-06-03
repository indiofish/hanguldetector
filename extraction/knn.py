from functools import partial
import numpy as np

def dist(ft1, ft2):
    """return the euclidian distance between the two feature vectors"""
    if len(ft1) == len(ft2):
        return np.linalg.norm(ft1 - ft2)
    else:
        # penalize heavily
        return 100

def aux(data, testft):
    feature, name, font, c = data
    return (dist(feature, testft), c)

def get_neighbors(trainset, testft, k):
    distances = [aux(d, testft) for d in trainset]
    distances = sorted(distances)

    return distances[:k]

def response(neighbors):
    votes = {}
    for (dist, label) in neighbors:
        if dist == 0:
            # some magic small number
            dist = 0.00000000001
        try:
            votes[label] += 1 / dist
        except KeyError:
            votes[label] = 1 / dist

    label = max(votes.items(), key=lambda x: x[1])[0]
    confidence = votes[label] / (sum(votes.values())) * 100

    return (label, confidence)
