"""
used for parts (a) through (d)
"""

from scipy.io import loadmat
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from k_means import k_means

if __name__ == "__main__":
    hw2data = loadmat("intro-to-machine-learning/homework 2/hw2data.mat")
    data = list([x[:2] for x in hw2data["clusterme"]])

    k = 3

    centroids, cluster_map = k_means(data=data, clusters=k, plot=[-1])

    print("Final centroids:")
    for i, centroid in centroids.items():
        print(i, centroid)
    print("Cluster sizes:", {i: len(indices) for i, indices in cluster_map.items()})

    ################################################################
    # for part d - gaussian mixture model
    gmm = GaussianMixture(n_components=k, means_init=list(centroids.values()), covariance_type='full', random_state=0)
    gmm.fit(data)

    responsibilities = gmm.predict_proba(data)
    log_likelihood = gmm.score(data) * len(data)

    print(f"{"Converged" if gmm.converged_ else "Did not converge"} after {gmm.n_iter_} iterations")
    print(f"Log-likelihood: {log_likelihood}")

    data_array = np.array(data)
    plt.figure(figsize=(8, 6))
    plt.scatter(data_array[:, 0], data_array[:, 1], c=responsibilities, s=40)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gaussian Mixture Responsibilities (RGB)")
    plt.show()
    