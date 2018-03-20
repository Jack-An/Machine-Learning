import numpy as np
import scipy.io as sio
from logistic_nn.alglib import display_data
from neural_network.alglib import nn_cost, sigmoid_gradient, rand_initialize_weights


def main():
    # setup parameters
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10

    # loading data and visualization
    print(' Loading and Visualizing Data ...')
    data = sio.loadmat('ex4data1.mat')
    X = data['X']
    y = data['y']
    # replace label 10 to 0
    # for i in range(y.size):
    #     if y[i] == 10:
    #         y[i] = 0
    m, n = X.shape
    sel = np.random.permutation(m)
    sel = sel[:100]
    display_data(X[sel, :])

    # loading parameters
    print(' Loading Saved Neural Network Parameters ...')
    parameters = sio.loadmat('ex4weights.mat')
    Theta1 = parameters['Theta1']
    Theta2 = parameters['Theta2']
    nn_parms = np.hstack((Theta1.flatten(), Theta2.flatten()))

    # Compute Cost (Feed forward)
    print('Feed forward Using Neural Network ...')
    lambd = 0
    J, grad = nn_cost(nn_parms, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)
    print('Cost at parameters (loaded from ex4weights): {:2.3f}'.format(J))

    # check reg cost
    print('nChecking Cost Function (w/ Regularization) ...')
    lambd = 1
    J, grad = nn_cost(nn_parms, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)
    print('Cost at parameters (loaded from ex4weights): {:2.3f}'.format(J))

    # check grad
    print('nEvaluating sigmoid gradient...')
    g = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
    print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:')
    print(g)

    # initializing parameters
    print('Initializing Neural Network Parameters ...')
    initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)

    # unroll parameters
    initial_nn_params = np.hstack((initial_Theta1.flatten(), initial_Theta2.flatten()))

    # implement Back propagation
    print('Checking Back propagation')


if __name__ == '__main__':
    main()
