"""
used for parts (a) through (d)
"""

from scipy.io import loadmat

from naive_bayes import naive_bayes

if __name__ == "__main__":
    hw2data = loadmat("intro-to-machine-learning/homework 2/hw2data.mat")
    data = hw2data["classifyme"][:, :5]
    labels = [int(x) for x in hw2data["classifyme"][:, -1]]

    specialx = hw2data["specialx_nb"][0]

    classifier = naive_bayes(data, labels)

    specialx_posteriors = dict(zip(classifier.classes_, classifier.predict_proba(specialx.reshape(1, -1))[0]))

    print(f"Posterior values for specialx ({specialx})")
    print(f"P(y = 1 | x) = {specialx_posteriors[1]:.6f}")
    print(f"P(y = 0 | x) = {specialx_posteriors[0]:.6f}")