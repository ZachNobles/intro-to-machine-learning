from scipy.io import loadmat
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    data = loadmat("intro-to-machine-learning/homework 3/hw3data.mat")
    data2 = data["data2"]

    features = data2[:, :2]
    labels = data2[:, 2]
    classes = np.unique(labels)

    X1 = features[labels == classes[0]]
    X2 = features[labels == classes[1]]

    # part a
    plt.scatter(X1[:, 0], X1[:, 1], marker="o")
    plt.scatter(X2[:, 0], X2[:, 1], marker="x")
    plt.title("Data2 Scatter Plot")
    plt.show()

    # part b
    mu1 = np.mean(X1, axis=0)
    mu2 = np.mean(X2, axis=0)

    X1_centered = X1 - mu1
    X2_centered = X2 - mu2

    S = X1_centered.T @ X1_centered + X2_centered.T @ X2_centered
    w = np.linalg.pinv(S) @ (mu1 - mu2)

    # part c
    w_unit = w / np.linalg.norm(w)
    print("Unit weight vector:", w_unit)
    X1_projected = X1 @ w_unit.T
    X2_projected = X2 @ w_unit.T

    plt.figure(figsize=(8, 5))
    plt.hist(X1_projected, bins=20, density=True, label="Class 1")
    plt.hist(X2_projected, bins=20, density=True, label="Class 2")

    plt.ylabel("Probability density")
    plt.title("Class Distributions After Projection")
    plt.legend()
    plt.show()

    # part d
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(X1.shape[0]), np.ones(X2.shape[0])))

    linear_model = SVC(kernel="linear")
    linear_model.fit(X, y)
    linear_accuracy = linear_model.score(X, y)
    print("Linear SVM accuracy:", linear_accuracy*100)

    radial_model = SVC(kernel="rbf")
    radial_model.fit(X, y)
    radial_accuracy = radial_model.score(X, y)
    print("Radial SVM accuracy:", radial_accuracy*100)