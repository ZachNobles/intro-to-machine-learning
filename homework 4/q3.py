from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import time

from common import get_data_sets


def part_a(X_train, X_test):
    print("Part a")
    t = time.perf_counter()
    pca_full = PCA(n_components=784).fit(X_train)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    C = int(np.argmax(cumulative_variance >= 0.95)) + 1
    pca = PCA(n_components=C, svd_solver="full").fit(X_train)
    Z_train = pca.transform(X_train)
    Z_test = pca.transform(X_test)
    pca_time = time.perf_counter() - t
    print(f"Number of components responsible for 95% variance: {C}")
    print(f"PCA fit + projection time: {pca_time:.2f} seconds\n")
    return Z_train, Z_test, pca_time


def part_b(Z_train, Y_train, Z_test, Y_test, pca_time):
    print("Part b")
    tree = DecisionTreeClassifier(random_state=0)

    t = time.perf_counter()
    tree.fit(Z_train, Y_train)
    fit_time = time.perf_counter() - t

    t = time.perf_counter()
    test_pred = tree.predict(Z_test)
    test_pred_time = time.perf_counter() - t

    train_acc = tree.score(Z_train, Y_train)
    test_acc = np.mean(test_pred == Y_test)
    print(f"Training accuracy: {train_acc * 100:.2f}%, test accuracy: {test_acc * 100:.2f}%")
    print(f"Nodes: {tree.tree_.node_count}, leaves: {tree.get_n_leaves()}, depth: {tree.get_depth()}")
    print(f"Tree fit time: {fit_time:.2f}s | test prediction time: {test_pred_time:.3f}s")
    print(f"Total training cost (PCA + tree fit): {pca_time + fit_time:.2f}s")


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_data_sets()
    Z_train, Z_test, pca_time = part_a(X_train, X_test)
    part_b(Z_train, Y_train, Z_test, Y_test, pca_time)