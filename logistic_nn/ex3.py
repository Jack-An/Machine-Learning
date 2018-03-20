import scipy.io as sio
import numpy as np
from logistic_nn.alglib import display_data, one_vs_all, predict
from logistic_regression.alglib import cost_reg, gradient_reg


def main():
    # Loading data
    print('Loading and Visualizing Data ...')
    mat_contents = sio.loadmat('ex3data1.mat')
    X = mat_contents['X']
    y = mat_contents['y']
    input_layer_size = 400
    num_labels = 10
    for i in range(y.size):
        if y[i] == 10:
            y[i] = 0
    # Visualizing Data
    m, n = X.shape
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]
    display_data(sel)

    # Test case for cost and gradient
    print('Testing lrCostFunction() with regularization')
    theta = np.array([-2, -1, 1, 2])
    X_t = (np.arange(1, 16).reshape(3, 5) / 10).T
    X_t = np.c_[np.ones(5), X_t]
    y_t = np.array([1, 0, 1, 0, 1]).reshape(5, 1)
    lambd_t = 3
    cost = cost_reg(theta, X_t, y_t, lambd_t)
    grad = gradient_reg(theta, X_t, y_t, lambd_t)
    print('Cost :{}'.format(cost))
    print('Gradients :{}'.format(grad))

    # One-vs-All Training
    lambd = 0.1
    all_theta = one_vs_all(X, y, num_labels, lambd)
    p = predict(all_theta, X)
    print('Train Accuracy: {0:.2f}'.format(np.mean(p == y) * 100))

    Predict
    r = np.random.permutation(m)
    X_temp = X[r, :]
    predictions = predict(all_theta, X_temp)
    for i in range(m):
        print('Displaying Example Image')
        display_data(X_temp[i, :])
        pred = predictions[i]
        print('Logistic Regression Prediction: {}'.format(pred))
        s = input('Paused - press enter to continue, q to exit:')
        if s == 'q':
            break


if __name__ == '__main__':
    main()
