import numpy as np

class GaussianNaiveBayes:
    def fit(self, data: np.ndarray, labels: np.ndarray):
        data = np.asarray(data, dtype=float)
        labels = np.asarray(labels)

        self.classes_, class_counts = np.unique(labels, return_counts=True)
        self.log_priors_ = np.log(class_counts / len(labels))
        self.means_ = np.empty((len(self.classes_), data.shape[1]))
        self.variances_ = np.empty_like(self.means_)

        for class_index, label in enumerate(self.classes_):
            class_data = data[labels == label]
            self.means_[class_index] = np.mean(class_data, axis=0)
            self.variances_[class_index] = np.maximum(np.var(class_data, axis=0), np.finfo(float).eps)

        training_error = np.mean(self.predict(data) != labels)
        print(f"Training error: {training_error:.2%}")
        return self

    def _joint_log_likelihood(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)

        log_likelihoods = np.empty((len(data), len(self.classes_)))
        for class_index in range(len(self.classes_)):
            centered = data - self.means_[class_index]
            log_likelihoods[:, class_index] = np.sum(
                -0.5 * (np.log(2 * np.pi * self.variances_[class_index])
                    + centered ** 2 / self.variances_[class_index]
                ), axis=1)

        return log_likelihoods + self.log_priors_

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        joint_log_likelihoods = self._joint_log_likelihood(data)
        joint_log_likelihoods -= np.max(joint_log_likelihoods, axis=1, keepdims=True)
        probabilities = np.exp(joint_log_likelihoods)
        return probabilities / np.sum(probabilities, axis=1, keepdims=True)

    def predict(self, data: np.ndarray) -> np.ndarray:
        joint_log_likelihoods = self._joint_log_likelihood(data)
        return self.classes_[np.argmax(joint_log_likelihoods, axis=1)]


def naive_bayes(data: np.ndarray, labels: np.ndarray) -> GaussianNaiveBayes:
    """Train and return a classifier that can predict new feature rows."""
    return GaussianNaiveBayes().fit(data, labels)