import math
import matplotlib.pyplot as plt
import numpy as np
from logistic_regression.alglib import sigmoid, cost_reg, gradient_reg
import scipy.optimize as op


def display_data(X, example_width=None):
    # DISPLAYDATA Display 2D data in a nice grid
    #   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    #   stored in X in a nice grid. It returns the figure handle h and the
    #   displayed array if requested.

    # closes previously opened figure. preventing a
    # warning after opening too many figures
    plt.close()

    # creates new figure
    plt.figure()

    # turns 1D X array into 2D
    if X.ndim == 1:
        X = np.reshape(X, (-1, X.shape[0]))

    # Set example_width automatically if not passed in
    if not example_width or not 'example_width' in locals():
        example_width = int(round(math.sqrt(X.shape[1])))

    # Gray Image
    plt.set_cmap("gray")

    # Compute rows, cols
    m, n = X.shape
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(math.floor(math.sqrt(m)))
    display_cols = int(math.ceil(m / display_rows))

    # Between images padding
    pad = 1

    # Setup blank display
    len1 = int(pad + display_rows * (example_height + pad))
    len2 = pad + display_cols * (example_width + pad)
    display_array = -np.ones((len1, len2))

    # Copy each example into a patch on the display array
    curr_ex = 1
    for j in range(1, display_rows + 1):
        for i in range(1, display_cols + 1):
            if curr_ex > m:
                break

            # Copy the patch

            # Get the max value of the patch to normalize all examples
            max_val = max(abs(X[curr_ex - 1, :]))
            rows = pad + (j - 1) * (example_height + pad) + np.array(range(example_height))
            cols = pad + (i - 1) * (example_width + pad) + np.array(range(example_width))

            # Basic (vs. advanced) indexing/slicing is necessary so that we look can assign
            # 	values directly to display_array and not to a copy of its subarray.
            # 	from stackoverflow.com/a/7960811/583834 and
            # 	bytes.com/topic/python/answers/759181-help-slicing-replacing-matrix-sections
            # Also notice the order="F" parameter on the reshape call - this is because python's
            #	default reshape function uses "C-like index order, with the last axis index
            #	changing fastest, back to the first axis index changing slowest" i.e.
            #	it first fills out the first row/the first index, then the second row, etc.
            #	matlab uses "Fortran-like index order, with the first index changing fastest,
            #	and the last index changing slowest" i.e. it first fills out the first column,
            #	then the second column, etc. This latter behaviour is what we want.
            #	Alternatively, we can keep the deault order="C" and then transpose the result
            #	from the reshape call.
            display_array[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] = np.reshape(X[curr_ex - 1, :],
                                                                                   (example_height, example_width),
                                                                                   order="F") / max_val
            curr_ex += 1

        if curr_ex > m:
            break

    # Display Image
    h = plt.imshow(display_array, vmin=-1, vmax=1)

    # Do not show axis
    plt.axis('off')

    plt.show()

    return h, display_array


def one_vs_all(X, y, num_labels, lambd):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.c_[np.ones(m), X]
    initial_theta = np.zeros(n + 1)
    for c in range(num_labels):
        result = op.minimize(fun=cost_reg,
                             x0=initial_theta,
                             args=(X, 1 * (y == c), lambd),
                             jac=gradient_reg,
                             method='TNC',
                             options={'maxiter': 100})
        all_theta[c, :] = result.x
    return all_theta


def predict(all_theta, X):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    return sigmoid(np.dot(X, all_theta.T)).argmax(axis=1).reshape(-1, 1)


def predict_nn(Theta1, Theta2, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    a2 = sigmoid(np.dot(X, Theta1.T))
    a2 = np.hstack((np.ones((m, 1)), a2))
    h_theta = sigmoid(np.dot(a2, Theta2.T))
    p = h_theta.argmax(axis=1).reshape(-1, 1)
    p += 1
    return p
