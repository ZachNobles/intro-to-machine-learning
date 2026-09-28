"""
used for parts (a) through (d)
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

from naive_bayes import naive_bayes

if __name__ == "__main__":
    hw2data = loadmat("intro-to-machine-learning/homework 2/hw2data.mat")
    data = hw2data["classifyme"][:, :2]
    labels = [int(x) for x in hw2data["classifyme"][:, -1]]

    close_point = data[3520] # this point is close to the boundary
    specialx = hw2data["specialx_nb"][0][:2]


    classifier = naive_bayes(data, labels)

    close_point_posteriors = dict(zip(classifier.classes_, classifier.predict_proba(close_point.reshape(1, -1))[0]))
    specialx_posteriors = dict(zip(classifier.classes_, classifier.predict_proba(specialx.reshape(1, -1))[0]))

    print(f"Posterior values for point 3520 ({close_point[0]}, {close_point[1]})")
    print(f"P(y = 1 | x) = {close_point_posteriors[1]:.6f}")
    print(f"P(y = 0 | x) = {close_point_posteriors[0]:.6f}")

    print(f"Posterior values for specialx ({specialx[0]}, {specialx[1]})")
    print(f"P(y = 1 | x) = {specialx_posteriors[1]:.6f}")
    print(f"P(y = 0 | x) = {specialx_posteriors[0]:.6f}")

    
    labels = np.asarray(labels)

    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min - x_padding, x_max + x_padding, 400),
        np.linspace(y_min - y_padding, y_max + y_padding, 400),
    )
    grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    predictions = classifier.predict(grid).reshape(x_grid.shape)

    class_cmap = plt.get_cmap("RdYlBu", 2)
    plt.contourf(x_grid, y_grid, predictions, levels=[-0.5, 0.5, 1.5], alpha=0.2, cmap=class_cmap)
    plt.contour(x_grid, y_grid, predictions, levels=[0.5], colors="black", linewidths=2)

    for label in range(2):
        points = data[labels == label]
        x, y = zip(*points)
        plt.scatter(x, y, color=class_cmap(label), label=f"Class {label}")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Naive Bayes Decision Boundary")
    plt.show()