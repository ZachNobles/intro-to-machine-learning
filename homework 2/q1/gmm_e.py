from scipy.io import loadmat
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
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
    D = len(data[0])
    N = len(data)

    scores = dict()

    for k in range(1, 9):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
        gmm.fit(data)

        responsibilities = gmm.predict_proba(data)
        log_likelihood = gmm.score(data) * len(data)

        print(f"(K={k}) {"Converged" if gmm.converged_ else "Did not converge"} after {gmm.n_iter_} iterations")
        print(f"Log-likelihood: {log_likelihood}\n")

        P = (k-1) + k*D + k * ((D*(D+1)) / 2)
        scores[k] = BIC(-log_likelihood, N, P)

    plt.scatter(scores.keys(), scores.values())
    plt.xlabel("K")
    plt.ylabel("BIC")
    plt.title("Bayesian Information Criterion Scores for K=1 through 8")
    plt.show()
