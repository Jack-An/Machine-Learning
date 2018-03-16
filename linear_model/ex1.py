import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linear_model.alglib import plot, plot_line, compute_cost, gradient_descent
from mpl_toolkits.mplot3d import Axes3D


def main():
    print(" Plotting Data ...")
    data = pd.read_csv('ex1data1.txt')
    df = pd.DataFrame(data)
    X = np.array(df.iloc[:, 0]).reshape(96, 1)
    y = np.array(df.iloc[:, 1]).reshape(96, 1)
    m = y.size
    plot(X, y)
    X = np.c_[np.ones(m), X]
    theta = np.zeros(2).reshape(2, 1)

    iterations = 1500
    alpha = 0.01

    print(" Testing the cost function ... \n")
    J = compute_cost(X, y, theta)
    print(" With theta = [0 ; 0]\nCost computed = {}\n".format(J))
    j2 = compute_cost(X, y, np.array([-1, 2]).reshape(2, 1))
    print(" With theta = [-1 ; 2]\nCost computed = {}\n".format(j2))

    theta, _ = gradient_descent(X, y, theta, alpha, iterations)
    print(" Theta found by gradient descent:")
    print(theta)
    plot_line(X, y, theta)

    # ========prediction========
    predict1 = np.array([1, 3.5]).dot(theta)
    print("For population = 35,000, we predict a profit of {}".format(predict1[0] * 10000))

    predict2 = np.array([1, 7]).dot(theta)
    print("For population = 70,000, we predict a profit of {}".format(predict2[0] * 10000))

    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

    for i in range(theta0_vals.size):
        for j in range(theta1_vals.size):
            t = np.array([[theta0_vals[i]],
                          [theta1_vals[j]]])
            J_vals[i][j] = compute_cost(X, y, t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(theta0_vals, theta1_vals)

    ax.plot_surface(X, Y, J_vals)

    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")
    ax.set_zlabel(r"$J$")
    plt.title('Cost with theta')
    plt.pause(5)
    plt.close()
    plt.contourf(X, Y, J_vals, locator=plt.LogLocator())
    plt.show()


if __name__ == '__main__':
    main()
