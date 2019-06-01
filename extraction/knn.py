from multiprocessing import Pool
from functools import partial
import score

def aux(data, testft):
    feature, name, font, c = data
    return (score.loss(feature, testft), c)

def get_neighbors(trainset, testft, k, THREADS=1):
    if THREADS > 1:
        p = Pool(THREADS)
        distances = p.map(partial(aux, testft=testft), trainset)
        distances = sorted(distances)
    else:
        distances = sorted([aux(d, testft) for d in trainset])

    return distances[:k]

def response(neighbors):
    votes = {}
    for (dist, label) in neighbors:
        try:
            votes[label] += 1 / dist
        except:
            votes[label] = 1 / dist
    label = max(votes.items(), key=lambda x: x[1])[0]

    return label
