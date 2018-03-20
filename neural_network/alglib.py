import numpy as np
import scipy.optimize as op


def nn_cost(nn_params,
            input_layer_size,
            hidden_layer_size,
            num_labels,
            X, y, lambd):
    Theta1 = np.reshape(nn_params[1])
    pass
