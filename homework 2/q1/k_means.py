import matplotlib.pyplot as plt
import numpy as np
import random

def plot_clusters(point_map: dict, cluster_map: dict, centroids: dict, error_sum: float, iteration: int):
    """
    Plots clusters. Only works for 2D points.
    """

    colors = ["b", "g", "r", "c", "m", "y"]
    for c in cluster_map.keys():
        members = np.array([point_map[i] for i in cluster_map[c]])
        x = members[:, 0]
        y = members[:, 1]
        plt.scatter(x, y, s=12, alpha=0.4, color=colors.pop(0), label=f"Cluster {c+1}")
        plt.plot(centroids[c][0], centroids[c][1], "X", ms=10, color="k")
    plt.suptitle(f"Clusters on Iteration {iteration+1}")
    plt.title(f"Sum of Squared Errors: {error_sum:.1f}")
    plt.legend()
    plt.show()

def distance(A, B):
    # You get 10 guesses for what this function does
    return np.linalg.norm(A - B) ** 2

def k_means(data: list, clusters: int, plot:list=[], max_iterations: int=200, tolerance=1e-6):
    """
    k means clustering
    clusters: list of data points
    clusters: integer representing number of clusters k
    plot: iterations to plot clusters on
    max_iterations: what it sounds like
    tolerance: the algorithm will stop upon the decrease in error being less than this threshold
    """

    points = np.asarray(data, dtype=float)

    if points.ndim == 1:
        points = points.reshape(-1, 1)

    centroids = {i: point.copy() for i, point in enumerate(random.sample(points.tolist(), clusters))}
    point_map = {i: point for i, point in enumerate(points)}

    last_error_sum = np.inf
    print("finding clusters")

    for iteration in range(max_iterations):
        error_sum = 0.0
        cluster_map = {c: [] for c in centroids.keys()}

        # assign each point to the nearest centroid
        for index, point in point_map.items():
            distances = {c: distance(point, centroid) for c, centroid in centroids.items()}
            closest_centroid = min(distances, key=distances.get)
            cluster_map[closest_centroid].append(index)
            error_sum += distances[closest_centroid]

        print(f"\rIteration {iteration + 1}: Error: {error_sum:.3f}", end="")

        if abs(last_error_sum - error_sum) < tolerance:
            break
        last_error_sum = error_sum

        # update centroids
        for c in centroids.keys():
            members = [point_map[i] for i in cluster_map[c]]
            if members:
                centroids[c] = np.mean(members, axis=0)

        # plot iterations in the list
        if (iteration+1) in plot:
            plot_clusters(point_map, cluster_map, centroids, error_sum, iteration)

    print()
    if -1 in plot:
        plot_clusters(point_map, cluster_map, centroids, error_sum, iteration)

    return centroids, cluster_map