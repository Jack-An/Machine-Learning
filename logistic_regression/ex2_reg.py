import numpy as np
import scipy.optimize as op
from logistic_regression.alglib import plot, map_feature, cost_reg, gradient_reg, plot_decision_boundary, predict
import matplotlib.pyplot as plt


def main():
    data = np.genfromtxt('ex2data2.txt', delimiter=',')
    X = data[:, [0, 1]]
    y = data[:, -1]
    pl, p1, p2 = plot(X, y)
    pl.xlabel('Microchip Test 1')
    pl.ylabel('Microchip Test 2')
    pl.legend((p1, p2), ('y=1', 'y=0'), numpoints=1, handlelength=0)
    pl.show()
    # Regularized Logistic Regression
    X = map_feature(X[:, 0], X[:, 1])

    initial_theta = np.zeros(X.shape[1])

    lambd = 1
    cost = cost_reg(initial_theta, X, y, lambd)
    grad = gradient_reg(initial_theta, X, y, lambd)
    print('Cost at initial theta (zeros): {}'.format(cost[0]))
    print('Gradient at initial theta (zeros) - first five values only:')
    print(grad[:5])
    test_theta = np.ones(X.shape[1])
    cost = cost_reg(test_theta, X, y, 10)
    grad = gradient_reg(test_theta, X, y, 10)
    print('Cost at test theta (with lambda = 10): {}'.format(cost[0]))
    print('Gradient at test theta  - first five values only:')
    print(grad[:5])

    initial_theta = np.zeros(X.shape[1])
    lambdas = [0, 1, 10, 100]
    for lambd in lambdas:
        # Regularization and Accuracies
        result = op.minimize(fun=cost_reg,
                             x0=initial_theta,
                             args=(X, y, lambd),
                             method='TNC',
                             jac=gradient_reg)
        optimal_theta = result.x
        plot_decision_boundary(optimal_theta, X, y, lambd)
        pl.show()
        p = predict(optimal_theta, X)
        m = y.size
        arr = 1 * (p == y.reshape(1, m))
        print('Train Accuracy (with lambda ={})  : {} %'.format(lambd, np.mean(arr) * 100))


if __name__ == '__main__':
    main()
