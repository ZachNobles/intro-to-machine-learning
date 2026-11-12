from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time

from common import get_data_sets

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_data_sets()

    model = KNeighborsClassifier(n_neighbors=4, n_jobs=-1)
    start_time = time.time()
    model.fit(X_train, Y_train)
    print(f"Fit time: {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    predictions = model.predict(X_test)
    print(f"Prediction time for test set: {time.time() - start_time:.2f} seconds")
    accuracy = np.mean(predictions == Y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    start_time = time.time()
    neighbor_indices = model.kneighbors(return_distance=False)
    neighbor_labels = Y_train[neighbor_indices]
    class_votes = np.stack([(neighbor_labels == label).sum(axis=1) for label in model.classes_], axis=1)
    predictions = model.classes_[np.argmax(class_votes, axis=1)]
    print(f"Prediction time for training set: {time.time() - start_time:.2f} seconds")
    accuracy = np.mean(predictions == Y_train)
    print(f"Training accuracy (excluding each point itself): {accuracy * 100:.2f}%")