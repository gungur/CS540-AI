import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


def plot_graph(data):
    x = data['year'].to_numpy(dtype=np.int64)
    y = data['days'].to_numpy(dtype=np.int64)
    plt.plot(x, y)
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.savefig('plot.jpg')


def create_data_matrix(data):
    x = data['year']
    x = x.to_numpy()
    X = np.empty(shape=(len(x), 2), dtype=np.int64)
    for i in range(len(x)):
        X[i] = [1, x[i]]
    print('Q3a:')
    print(X)

    y = data['days']
    Y = y.to_numpy(dtype=np.int64)
    print('Q3b:')
    print(Y)
    return X, Y


def matrix_computations(X, Y):
    Z = np.dot(np.transpose(X), X)
    print('Q3c:')
    print(Z)

    I = np.linalg.inv(Z)
    print('Q3d:')
    print(I)

    PI = np.dot(I, np.transpose(X))
    print('Q3e:')
    print(PI)

    B = np.dot(PI, Y)
    print('Q3f:')
    print(B)
    return B


def prediction(B):
    x_test = 2022
    y_test = B[0] + (B[1] * x_test)
    print('Q4: ' + str(y_test))

    sign = np.sign(B[1])
    if sign == -1:
        print('Q5a: ' + str('<'))
    elif sign == 0:
        print('Q5a: ' + str('='))
    elif sign == 1:
        print('Q5a: ' + str('>'))

    print('Q5b: ' + str('The linear regression model calculated is, as the name suggests, a linear equation. B_0 is '
                        'the y-intercept while B_1 is the slope. If the sign of B_1 is positive (>), it means that '
                        'over time, the number of days of ice cover increases. If the sign is zero (=), it means the '
                        'days of ice cover is constant at B_0. If the sign is negative (<), like in this case, '
                        'it means that over time, the number of days of ice cover over Lake Mendota decreases.'))

    x_star = -B[0] / B[1]
    print('Q6a: ' + str(x_star))
    print('Q6b: ' + str('I believe x* is a compelling prediction based on the trends in the data. Looking at the '
                        'graph plotted using real data, in the span of 166 years (1855-2021), the days of ice cover '
                        'only went down by 33 days (188 to 85). Thus, it is a compelling prediction that in 434-435 '
                        'years, Lake Mendota will no longer freeze.'))


def main():
    data = pd.read_csv(sys.argv[1])
    plot_graph(data)
    X, Y = create_data_matrix(data)
    B = matrix_computations(X, Y)
    prediction(B)


if __name__ == '__main__':
    main()
