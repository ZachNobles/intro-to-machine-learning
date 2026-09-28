from scipy.io import loadmat
import numpy as np
from sklearn.mixture import GaussianMixture
import math

from k_means import k_means

def BIC(L: float, N: int, P: int):
    """
    Bayesian Information Criterion
    L: negative log likelihood
    N: number of data points
    P: number of parameters in the GMM
    """
    return 2 * L + P * math.log(N)

if __name__ == "__main__":
    hw2data = loadmat("intro-to-machine-learning/homework 2/hw2data.mat")
    data = list(hw2data["clusterme"])
    specialx = hw2data["specialx"]

    k = 4
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
    gmm.fit(data)

    responsibilities = gmm.predict_proba(data)
    log_likelihood = gmm.score(data) * len(data)

    print(f"{"Converged" if gmm.converged_ else "Did not converge"} after {gmm.n_iter_} iterations")
    print(f"Log-likelihood: {log_likelihood}\n")


    responsibilities = list(*gmm.predict_proba(specialx))
    max_index, max_value = max(enumerate(responsibilities), key=lambda x: x[1])
    print(f"max responsibility: cluster {max_index+1} with mean {max_value}")