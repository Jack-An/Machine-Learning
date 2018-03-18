import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linear_model.alglib import plot, plot_line, compute_cost, gradient_descent, plot_surf, plot_contour

from matplotlib import cm


def main():
    print(' Plotting Data ...')
    data = pd.read_csv('ex1data1.txt')
    df = pd.DataFrame(data)
    X = np.array(df.iloc[:, 0]).reshape(96, 1)
    y = np.array(df.iloc[:, 1]).reshape(96, 1)
    m = y.size
    pl, p1 = plot(X, y)
    pl.show()
    X = np.c_[np.ones(m), X]
    theta = np.zeros(2).reshape(2, 1)

    iterations = 1500
    alpha = 0.01

    print('Testing the cost function ...')
    J = compute_cost(X, y, theta)
    print(' With theta = [0 ; 0]\nCost computed = {}'.format(J))
    j2 = compute_cost(X, y, np.array([-1, 2]).reshape(2, 1))
    print('With theta = [-1 ; 2]\nCost computed = {}'.format(j2))

    theta, _ = gradient_descent(X, y, theta, alpha, iterations)
    print(' Theta found by gradient descent:')
    print(theta)
    plot_line(X, y, theta)

    # ========prediction========
    predict1 = np.array([1, 3.5]).dot(theta)
    print('For population = 35,000, we predict a profit of {}'.format(predict1[0] * 10000))

    predict2 = np.array([1, 7]).dot(theta)
    print('For population = 70,000, we predict a profit of {}'.format(predict2[0] * 10000))
    # plot surface
    theta0_vals, theta1_vals, J_vals = plot_surf(X, y)
    # contour plot
    plot_contour(theta, theta0_vals, theta1_vals, J_vals)


if __name__ == '__main__':
    main()
