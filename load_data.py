from scipy.io import loadmat
if __name__ == "__main__":
    data = loadmat("intro-to-machine-learning/homework 2/hw2data.mat")
    print(list(data.keys()))