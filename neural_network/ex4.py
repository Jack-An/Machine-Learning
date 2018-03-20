import numpy as np
import scipy.io as sio
from logistic_nn.alglib import display_data


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
    nn_parms = np.hstack((Theta1.flatten(), Theta2.faltten()))

    # Compute Cost (Feed forward)
    print('Feed forward Using Neural Network ...')
    lambd =0


if __name__ == '__main__':
    main()
