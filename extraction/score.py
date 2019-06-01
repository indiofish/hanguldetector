import numpy as np

def loss(ft1, ft2):
    """return the euclidian distance between the two feature vectors"""
    if len(ft1) == len(ft2):
        return np.linalg.norm(ft1 - ft2)
    else:
        # penalize heavily
        return 100
