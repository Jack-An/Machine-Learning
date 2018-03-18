import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot(X, y):
    p1 = plt.scatter(X, y, marker='x', color='r')
    return plt, p1


def plot_line(X, y, theta):
    pl, p1 = plot(X[:, 1], y)
    pl.plot(X[:, 1], X.dot(theta), color='blue')
    pl.show()


def plot_surf(X, y):
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

    for i in range(theta0_vals.size):
        for j in range(theta1_vals.size):
            t = np.array([[theta0_vals[i]],
                          [theta1_vals[j]]])
            J_vals[i][j] = compute_cost(X, y, t)
    J_vals = J_vals.T
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
    surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap=cm.coolwarm, rstride=2)
    fig.colorbar(surf)
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.show()
    return theta0_vals, theta1_vals, J_vals


def plot_contour(theta, theta0_vals, theta1_vals, J_vals):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cset = plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20), cmap=cm.coolwarm)
    fig.colorbar(cset)
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.plot(theta[0, 0], theta[1, 0], 'rx', markersize=10, linewidth=2)
    plt.show()


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
