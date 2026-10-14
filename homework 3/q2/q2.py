from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


if __name__ == "__main__":
    data = loadmat("intro-to-machine-learning/homework 3/hw3data.mat")
    data3 = data["data3"]

    features = data3[:, :2]
    labels = data3[:, 2]
    classes = np.unique(labels)

    X1 = features[labels == classes[0]]
    X2 = features[labels == classes[1]]
    X3 = features[labels == classes[2]]

    # part a
    plt.scatter(X1[:, 0], X1[:, 1], color="blue", marker="o")
    plt.scatter(X2[:, 0], X2[:, 1], color="orange",marker="x")
    plt.scatter(X3[:, 0], X3[:, 1], color="teal", marker="s")
    plt.title("Data3 Scatter Plot")
    plt.show()

    # part b
    model = LogisticRegression()
    model.fit(features, labels)
    accuracy = model.score(features, labels)
    probabilities = model.predict_proba(features)
    class_indices = np.searchsorted(model.classes_, labels)
    cross_entropy_loss = -np.mean(np.log(probabilities[np.arange(len(labels)), class_indices]))
    print(f"Cross-entropy loss: {cross_entropy_loss:.3f}")
    print(f"Logistic Regression accuracy: {accuracy*100:.3f}")
    print(f"Error: {(1 - accuracy)*100:.3f}")

    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    predictions = model.predict(np.c_[xx.ravel(), yy.ravel()])
    predictions = predictions.reshape(xx.shape)
    plt.contourf(xx, yy, predictions, alpha=0.2)
    plt.scatter(X1[:, 0], X1[:, 1], color="blue", marker="o")
    plt.scatter(X2[:, 0], X2[:, 1], color="orange",marker="x")
    plt.scatter(X3[:, 0], X3[:, 1], color="teal", marker="s")
    plt.title("Logistic Regression Decision Boundaries")
    plt.show()

    # part c
    furthest_index = np.argmin(probabilities[np.arange(len(labels)), class_indices])
    furthest_point = features[furthest_index]
    print(f"Furthest point index: {furthest_index}")

    ambiguous_index = np.argmin(np.max(probabilities, axis=1))
    ambiguous_point = features[ambiguous_index]
    print(f"Most ambiguous point index: {ambiguous_index}")

    plt.scatter(X1[:, 0], X1[:, 1], color="blue", marker="o")
    plt.scatter(X2[:, 0], X2[:, 1], color="orange",marker="x")
    plt.scatter(X3[:, 0], X3[:, 1], color="teal", marker="s")
    plt.scatter(furthest_point[0], furthest_point[1], marker="*", color="red", s=200)
    plt.scatter(ambiguous_point[0], ambiguous_point[1], marker="*", color="springgreen", s=200)
    plt.title("Data3 Scatter Plot")
    plt.show()

    # part d
    model = QuadraticDiscriminantAnalysis()
    model.fit(features, labels)

    probabilities = model.predict_proba(features)
    cross_entropy_loss = -np.mean(np.log(probabilities[np.arange(len(labels)), class_indices]))
    print(f"\nCross-entropy loss (QDA): {cross_entropy_loss:.3f}")
    accuracy = model.score(features, labels)
    print(f"QDA accuracy: {accuracy*100:.3f}")
    print(f"Error: {(1 - accuracy)*100:.3f}")

    qda_predictions = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, qda_predictions, alpha=0.2)
    plt.scatter(X1[:, 0], X1[:, 1], color="blue", marker="o")
    plt.scatter(X2[:, 0], X2[:, 1], color="orange", marker="x")
    plt.scatter(X3[:, 0], X3[:, 1], color="teal", marker="s")
    plt.title("Quadratic Discriminant Analysis Decision Boundaries")
    plt.show()
