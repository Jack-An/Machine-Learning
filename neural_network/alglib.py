import numpy as np
import scipy.optimize as op
from logistic_regression.alglib import sigmoid


def sigmoid_gradient(z):
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


def nn_cost(nn_params,
            input_layer_size,
            hidden_layer_size,
            num_labels,
            X, y, lambd):
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1))
    m, n = X.shape
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    X = np.c_[np.ones(m), X]
    a2 = sigmoid(X.dot(Theta1.T))
    a2 = np.c_[np.ones(m), a2]
    h_theta = sigmoid(a2.dot(Theta2.T))
    yk = np.zeros((m, num_labels))
    # in python vector[-1] = vector[length -1]
    # as we changed label '10' to '0', so we should change back again in vectorization
    for i in range(m):
        yk[i, y[i] - 1] = 1
    # as the formula
    J = (-1 / m) * np.sum(np.sum(yk * np.log(h_theta) + (1 - yk) * np.log(1 - h_theta)))
    lambda_part = lambd * (np.sum(np.sum(Theta1[:, 1:] ** 2)) + np.sum(Theta2[:, 1:] ** 2)) / (2 * m)
    J = J + lambda_part
    """
    It's really difficult to understand the algorithm .
    This is second time I finished the algorithm, but I still 
    feel very hard to do this quickly. When I do this , I write down 
    all the matrix' size , and follow the formula and then understand what 
    I'm doing now . I think it's a useful way to help beginners to understand the 
    steps of the algorithm. 
    """
    # back prop
    for i in range(m):
        # forward prop
        a1 = X[i, :].reshape(1, -1)
        z2 = a1.dot(Theta1.T)
        a2 = sigmoid(z2)
        a2 = np.c_[np.ones(1), a2]
        z3 = a2.dot(Theta2.T)
        a3 = sigmoid(z3)

        # back prop
        z2 = np.c_[np.ones(1), z2]
        delta_3 = a3 - yk[i, :]

        delta_2 = (delta_3.dot(Theta2)) * sigmoid_gradient(z2)
        delta_2 = delta_2[:, 1:]
        Theta2_grad = Theta2_grad + delta_3.T.dot(a2)
        Theta1_grad = Theta1_grad + delta_2.T.dot(a1)

    # set regularized part
    Theta1_grad[:, 0] = Theta1_grad[:, 0] / m
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] / m + ((lambd / m) * Theta1[:, 1:])
    Theta2_grad[:, 0] = Theta2_grad[:, 0] / m
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] / m + ((lambd / m) * Theta2[:, 1:])
    # vectorization the Theta1_grad & Theta2_grad
    grad = np.hstack((Theta1_grad.flatten(), Theta2_grad.flatten()))
    return J, grad


def rand_initialize_weights(L_in, L_out):
    """
    This is very import for training the neural network ,
    and Andrew Ng has analysed on the class.
    :param L_in:
    :param L_out:
    :return W :
    """
    epsilon_init = 0.12
    W = np.random.rand(L_out, L_in + 1) * 2 * epsilon_init - epsilon_init
    return W
