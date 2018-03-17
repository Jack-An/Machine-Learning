import numpy as np
import matplotlib.pyplot as plt


def plot(X, y, label1, label2, x_label, y_label):
    pos = np.nonzero(y == 1)[0]
    neg = np.nonzero(y == 0)[0]
    a = X[pos, :]
    b = X[neg, :]
    plt.scatter(a[:, 0], a[:, 1], marker='+', s=15, label=label1)
    plt.scatter(b[:, 0], b[:, 1], marker='o', s=15, label=label2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
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


def cost_reg(theta, X, y, lambd):
    J = cost(theta, X, y)
    m, n = X.shape
    temp = theta.copy()
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
    '''
    Maps the two input features to quadratic features.
    Returns a new feature array with more features, comprising of
    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
    Inputs X1, X2 must be the same size
    '''
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    out = np.ones(shape=(x1[:, 0].size, 1))

    m, n = out.shape

    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = np.append(out, r, axis=1)

    return out


def plot_boundary(theta):
    data = np.genfromtxt('ex2data2.txt', delimiter=',')
    X = data[:, [0, 1]]
    y = data[:, -1]
    pos = np.nonzero(y == 1)[0]
    neg = np.nonzero(y == 0)[0]
    a = X[pos, :]
    b = X[neg, :]
    plt.scatter(a[:, 0], a[:, 1], marker='+', s=15, label='y=1')
    plt.scatter(b[:, 0], b[:, 1], marker='o', s=15, label='y=0')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    x1 = np.linspace(-1, 1.5, 50)
    x2 = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            feature = map_feature(np.array([x1[i]]), np.array([x2[j]]))
            z[i][j] = np.dot(theta, feature.T)
    z = z.T
    u, v = np.meshgrid(x1, x2)
    plt.contour(x1, x2, z, [0])
    plt.show()
