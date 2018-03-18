import numpy as np
import matplotlib.pyplot as plt


def plot(X, y):
    pos = np.nonzero(y == 1)[0]
    neg = np.nonzero(y == 0)[0]
    a = X[pos, :]
    b = X[neg, :]
    p1 = plt.scatter(a[:, 0], a[:, 1], marker='+', s=25)
    p2 = plt.scatter(b[:, 0], b[:, 1], marker='o', s=25)
    return plt, p1, p2


def plot_decision_boundary(theta, X, y, lambd=0):
    pl, p1, p2 = plot(X[:, 1:3], y)
    if X.shape[1] <= 3:
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
        p3 = plt.plot(plot_x, plot_y, color='blue', label='Decision Boundary')[0]
        pl.legend((p1, p2, p3), ('Admitted', 'Not Admitted', 'Decision Boundary'))
        pl.axis([30, 100, 30, 100])
        pl.show()
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i][j] = np.dot(map_feature(np.array([[u[i]]]), np.array([[v[j]]])), theta)
        z = z.T
        pl.contour(u, v, z, [0])
        pl.annotate('With lambda = {}'.format(lambd),
                    xy=(0.25, 0.25), xytext=(0.5, 0.5))
        pl.legend((p1, p2), ('y = 1', 'y = 0'), numpoints=1, handlelength=0)
        plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# the order of the args must be theta,X,y
def cost(theta, X, y):
    m, n = X.shape
    theta = theta.reshape(n, 1)
    prediction = sigmoid(X.dot(theta))
    left = -(y.T.dot(np.log(prediction)))
    right = (1 - y).T.dot(np.log(1 - prediction))
    J = (1 / m) * (left - right)
    return J


def cost_reg(theta, X, y, lambd):
    J = cost(theta, X, y)
    m, n = X.shape
    temp = theta.copy()
    temp[0] = 0
    lambda_part = (lambd / (2 * m)) * sum(np.square(temp))
    return J + lambda_part


def gradient(theta, X, y):
    m, n = X.shape
    y = y.reshape(m, 1)
    theta = theta.reshape(n, 1)
    prediction = sigmoid(X.dot(theta))
    grad = (1.0 / m) * (X.transpose().dot(prediction - y))
    return grad.flatten()


def gradient_reg(theta, X, y, lambd):
    theta = theta.reshape(theta.size, 1)
    m = y.size
    pred = sigmoid(X.dot(theta))
    grad = np.dot(X.T, pred - y.reshape(m, 1)) / m
    grad[1:] = grad[1:] + lambd * theta[1:] / m
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


def map_feature(x1, x2):
    """
    Maps the two input features to quadratic features.
    Returns a new feature array with more features, comprising of
    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
    Inputs X1, X2 must be the same size
    """
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    out = np.ones(shape=(x1[:, 0].size, 1))

    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = np.append(out, r, axis=1)

    return out
