from numpy.linalg.linalg import norm
import numpy as np


def obj_func(thetas, predicted_price_change, actual_price_change):
    return norm(
        np.subtract(
            np.transpose(actual_price_change),
            np.add(thetas[0],
                   np.dot(thetas[1:len(thetas)],
                          np.transpose(predicted_price_change)))))
