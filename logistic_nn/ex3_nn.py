import numpy as np
import scipy.io as sio
from logistic_nn.alglib import display_data, predict_nn


def main():
    # Load data
    print('Loading and Visualizing Data ...')
    mat_contents = sio.loadmat('ex3data1.mat')
    X = mat_contents['X']
    y = mat_contents['y']

    # set up the parameters , may be used
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10

    # Visualizing Data
    m, n = X.shape
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]
    display_data(sel)

    # Load parameters
    print('Loading Saved Neural Network Parameters ...')
    data = sio.loadmat('ex3weights.mat')
    Theta1 = data['Theta1']
    Theta2 = data['Theta2']
    pred = predict_nn(Theta1, Theta2, X)
    print('Training Set Accuracy: {0:.2f} %'.format(np.mean(pred == y) * 100))

    rp = np.random.permutation(m)
    # print(rp.shape)
    for i in range(m):
        print('Displaying Example Images')
        display_data(X[rp[i], :].reshape(1, -1))
        pred = predict_nn(Theta1, Theta2, X[rp[i], :].reshape(1, -1))
        print('Neural Network Prediction : {:d}'.format(np.mod(pred[0, 0], 10)))
        s = input('Paused - press enter to continue, q to exit:')
        if s == 'q':
            break


if __name__ == '__main__':
    main()
