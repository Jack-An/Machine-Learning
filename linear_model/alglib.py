import matplotlib.pyplot as plt
import numpy as np


def plot(x, y):
    plt.scatter(x, y, marker='x', color='r')
    plt.pause(5)
    plt.close()


def plot_line(X, y, theta):
    plt.scatter(X[:, 1], y, marker='x', color='r')
    plt.plot(X[:, 1], X.dot(theta), color='blue')
    plt.pause(5)
    plt.close()


def compute_cost(X, y, theta):
    m = y.size
    prediction = X.dot(theta)
    sqr_error = (prediction - y) ** 2
    J = (1 / (2 * m)) * np.sum(sqr_error)
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros(num_iters).reshape(num_iters, 1)
    for i in range(num_iters):
        theta = theta - (alpha / m) * (X.transpose().dot(X.dot(theta) - y))
        J_history[i, 0] = compute_cost(X, y, theta)
    return theta, J_history


def feature_normalize(X):
    X_norm = X
    _, t = X.shape
    mu = np.zeros((1, t))
    sigma = np.zeros((1, t))
    for i in range(t):
        mu[0][i] = np.mean(X[:, i])
        sigma[0][i] = np.std(X[:, i])
        X_norm[:, i] = (X_norm[:, i] - mu[0][i]) / sigma[0][i]

    return X_norm, mu, sigma


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        theta = theta - (alpha / m) * (X.transpose().dot(X.dot(theta) - y))
        J_history[i][0] = compute_cost(X, y, theta)
    return theta, J_history


def normal_eqn(X, y):
    theta = np.linalg.pinv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    return theta
