import numpy as np

def loss(ft1, ft2):
    """return the euclidian distance between the two feature vectors"""
    # if len(ft1) < len(ft2):
        #penalize those with different number of boxes?
        # ft1 = np.pad(ft1, (0, len(ft2)-len(ft1)), 'constant',
                # constant_values=1)
    # elif len(ft1) > len(ft2):
        # ft2 = np.pad(ft2, (0, len(ft1)-len(ft2)), 'constant',
                # constant_values=1)
    if len(ft1) == len(ft2):
        return np.linalg.norm(ft1 - ft2)
    else:
        return 100

