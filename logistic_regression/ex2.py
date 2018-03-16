import numpy as np


def main():
    data = np.genfromtxt('ex2data1.txt', delimiter=',')
    m, _ = data.shape
    X = data[:, 0:2].reshape(m, 2)
    y = data[:, 2].reshape(m, 1)


if __name__ == '__main__':
    main()
