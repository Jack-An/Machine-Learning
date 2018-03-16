import numpy as np
import matplotlib.pyplot as plt
from linear_model.alglib import feature_normalize, gradient_descent, normal_eqn


def main():
    # =========Loading Dataset=======
    print(" Plotting Data ...")
    data = np.genfromtxt('ex1data2.txt', delimiter=',')
    X = data[:, :2].reshape(47, 2)
    y = data[:, -1].reshape(47, 1)
    m = y.size

    # Normalizing Features
    X, mu, sigma = feature_normalize(X)
    X = np.c_[np.ones(m), X]

    # Gradient descent
    alpha = 0.01
    num_iters = 400

    theta = np.zeros((3, 1))
    theta, his = gradient_descent(X, y, theta, alpha, num_iters)
    x_axis = np.arange(his.size)
    plt.plot(x_axis, his)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.pause(5)
    plt.close()
    print(' Theta computed from gradient descent: ')
    print(theta)

    # prediction
    price = np.array([1, (1650 - mu[0][0]) / sigma[0][0], (3 - mu[0][1]) / sigma[0][1]]).dot(theta)
    print('Predicted price of a 1650 sq-ft, 3 br house '
          '(using gradient descent):\n $ {}\n'.format(price[0]))

    # normal equation

    print(' Solving with normal equations...')
    data = np.genfromtxt('ex1data2.txt', delimiter=',')
    X = data[:, :2].reshape(47, 2)
    y = data[:, -1].reshape(47, 1)
    m = y.size
    X = np.c_[np.ones(m), X]
    theta = normal_eqn(X, y)
    print('Theta computed from the normal equations: ')
    print(theta)

    # prediction
    price = np.array([1, 1650, 3]).dot(theta)
    print('Predicted price of a 1650 sq-ft, 3 br house '
          '(using normal equations):\n ${}'.format(price[0]))


if __name__ == '__main__':
    main()
