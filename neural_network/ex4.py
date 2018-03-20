import numpy as np
import scipy.io as sio
from logistic_nn.alglib import display_data
from neural_network.alglib import nn_cost, sigmoid_gradient, rand_initialize_weights, check_nn_gradients, predict
import scipy.optimize as op


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
    for i in range(y.size):
        if y[i] == 10:
            y[i] = 0
    m, n = X.shape
    sel = np.random.permutation(m)
    sel = sel[:100]
    display_data(X[sel, :])

    # loading parameters
    print(' Loading Saved Neural Network Parameters ...')
    parameters = sio.loadmat('ex4weights.mat')
    Theta1 = parameters['Theta1']
    Theta2 = parameters['Theta2']
    nn_params = np.hstack((Theta1.flatten(), Theta2.flatten()))

    # Compute Cost (Feed forward)
    print('Feed forward Using Neural Network ...')
    lambd = 0
    J, grad = nn_cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)
    print('Cost at parameters (loaded from ex4weights): {:2.3f}'.format(J))

    # check reg cost

    """
    cost I used Andrew Ng trained parameters in Matlab ,as we changed the label "10" to "0" ,
    so the cost J is larger than we see in the script of Matlab)
    """
    print('nChecking Cost Function (w/ Regularization) ...')
    lambd = 1
    J, grad = nn_cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)
    print('Cost at parameters (loaded from ex4weights): {:2.3f} '.format(J))

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
    check_nn_gradients()

    lambd = 3
    check_nn_gradients(lambd)
    debug_J, _ = nn_cost(nn_params, input_layer_size,
                         hidden_layer_size,
                         num_labels,
                         X, y, lambd)
    print('Cost at (fixed) debugging parameters (w/ lambda = {}): {} \n'.format(lambd, debug_J))

    # Training NN
    print('Training Neural Network... ')
    lambd = 1
    cost_fun = lambda p: nn_cost(p,
                                 input_layer_size,
                                 hidden_layer_size,
                                 num_labels,
                                 X, y, lambd)
    result = op.minimize(fun=cost_fun,
                         x0=initial_nn_params,
                         args=(),
                         jac=True,
                         method='TNC',
                         options={'maxiter': 100})
    theta = result.x
    Theta1 = np.reshape(theta[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(theta[Theta1.size:], (num_labels, hidden_layer_size + 1))

    display_data(Theta1[:, 1:])
    p = predict(Theta1, Theta2, X)
    print('Training Set Accuracy: {}'.format(np.mean(p == y) * 100))
    cost, _ = nn_cost(theta, input_layer_size, hidden_layer_size, num_labels, X, y, 3)
    print('Cost at trained  parameters (w/ lambda = {}): {} \n'.format(3, cost))
    rp = np.random.permutation(m)
    for i in range(m):
        print('Displaying Example Images')
        display_data(X[rp[i], :].reshape(1, -1))
        pred = predict(Theta1, Theta2, X[rp[i], :].reshape(1, -1))
        print('Neural Network Prediction : {:d}'.format(pred[0][0]))
        s = input('Paused - press enter to continue, q to exit:')
        if s == 'q':
            break


if __name__ == '__main__':
    main()
