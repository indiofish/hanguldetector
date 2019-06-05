import numpy as np
import heapq

def dist(ft1, ft2):
    """return the euclidian distance between the two feature vectors"""
    if ft1.size == ft2.size:
        return np.linalg.norm(ft1 - ft2)
    else:
        # penalize heavily
        return 100

def get_neighbors(trainset, testft, k):
    # feature, name, font, c = d in trainset
    distances = ((dist(d[0], testft), d[3]) for d in trainset)
    # distances = sorted(distances)
    return heapq.nsmallest(k, distances)

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
