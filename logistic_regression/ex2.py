import numpy as np
from logistic_regression.alglib import plot, cost, gradient, plot_decision_boundary, sigmoid, predict
import scipy.optimize as op


def main():
    data = np.genfromtxt('ex2data1.txt', delimiter=',')
    m, _ = data.shape
    X = data[:, 0:2].reshape(m, 2)
    y = data[:, 2].reshape(m, 1)
    plot(X, y, 'Admitted', 'Not admitted', 'Exam 1 score', 'Exam 2 score')
    m, n = X.shape
    X = np.c_[np.ones(m), X]
    initial_theta = np.zeros(X.shape[1])
    print('Cost at initial theta (zeros): {}'.format(cost(initial_theta, X, y)[0][0]))
    print('Gradient at initial theta (zeros):')
    print(gradient(initial_theta, X, y))
    test_theta = np.array([-24, 0.2, 0.2])
    print('Cost at initial theta (zeros): {}'.format(cost(test_theta, X, y)[0][0]))
    print('Gradient at initial theta (zeros):')
    print(gradient(test_theta, X, y))
    result = op.minimize(fun=cost, x0=initial_theta,
                         args=(X, y),
                         method='TNC',
                         jac=gradient)
    optimal_theta = result.x
    optimal_cost = result.fun
    print('Cost at theta found by minimize:')
    print(optimal_cost)
    print('theta:')
    print(optimal_theta)
    plot_decision_boundary(optimal_theta, X, y)

    # predict and Accuracies
    prob = sigmoid(np.array([1, 45, 85]).dot(optimal_theta))
    print('For a student with scores 45 and 85, we predict an admission '
          'probability of {} '.format(prob))
    p = predict(optimal_theta, X)
    arr = 1 * (p == y.reshape(1, m))

    print('Train Accuracy: {} %'.format(np.mean(arr) * 100))


if __name__ == '__main__':
    main()
