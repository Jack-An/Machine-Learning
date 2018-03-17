import numpy as np
import matplotlib.pyplot as plt


def plot(X, y):
    pos = np.nonzero(y == 1)[0]
    neg = np.nonzero(y == 0)[0]
    a = X[pos, :]
    b = X[neg, :]
    plt.scatter(a[:, 0], a[:, 1], marker='+', s=15, label='Admitted')
    plt.scatter(b[:, 0], b[:, 1], marker='o', s=15, label='Not admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.pause(5)
    plt.close()


def plot_decision_boundary(theta, X, y):
    X = X[:, 1:]
    pos = np.nonzero(y == 1)[0]
    neg = np.nonzero(y == 0)[0]
    a = X[pos, :]
    b = X[neg, :]
    plot_x = np.array([np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2])
    plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
    plt.scatter(a[:, 0], a[:, 1], marker='+', s=15, label='Admitted')
    plt.scatter(b[:, 0], b[:, 1], marker='o', s=15, label='Not admitted')
    plt.plot(plot_x, plot_y, color='blue', label='Decision Boundary')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.pause(5)
    plt.close()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# the order of the args must be theta,X,y
def cost(theta, X, y):
    m, n = X.shape
    theta = theta.reshape(n, 1)
    prediction = sigmoid(X.dot(theta))
    left = -(y.transpose().dot(np.log(prediction)))
    right = ((1 - y).transpose()).dot(np.log(1 - prediction))
    J = (1 / m) * (left - right)
    return J


def gradient(theta, X, y):
    m, n = X.shape
    theta = theta.reshape(n, 1)
    prediction = sigmoid(X.dot(theta))
    grad = ((1 / m) * (X.transpose().dot(prediction - y)))
    return grad


def predict(theta, X):
    m = X.shape[0]
    p = sigmoid(X.dot(theta))
    for i in range(m):
        if p[i] < 0.5:
            p[i] = 0
        else:
            p[i] = 1
    return p
