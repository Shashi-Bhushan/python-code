import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit

# Fixing random state for reproducibility
np.random.seed(100)


def get_linear_points():
    # X is range 0 to 50
    x = np.arange(0, 100, 2)
    # Y is X +/- a random number
    y = x + (np.random.rand(x.shape[0]) * 10)

    return x, y


def linear_plot():
    x, y = get_linear_points()

    intercept, slope = polyfit(x, y, 1)

    plt.figure(1)
    # Plot Actual points
    plt.plot(x, y, '.')

    # Plot Predicted Values
    plt.plot(x, intercept + slope * x, '-')

    plt.text(2, 50, 'Linearity Present')
    plt.xlabel('Independent Variable(x)')
    plt.ylabel('Dependent Variable(y)')


def non_linear_plot():
    plt.figure(2)
    # X is range 0 to 50
    x = np.arange(0, 100, 2)
    # Y is X^3 +/- a random number
    y = x + ( np.random.rand(x.shape[0]) / 100 - (x ** 3))

    # Because of a cubic function, x and y no longer share a linear relation

    plt.plot(x, y, '.')
    plt.text(2, 2000, 'Linearity Absent')
    plt.xlabel('Independent Variable(x)')
    plt.ylabel('Dependent Variable(y)')


def linear_plot_residual():
    x, y = get_linear_points()

    intercept, slope = polyfit(x, y, 1)

    predicted = (intercept + slope * x)

    residual = y - predicted

    plt.figure(1)
    # Plot Observed points
    plt.plot(x, y, '.')

    # Plot Predicted Values
    plt.plot(x, predicted, '-')

    # Plot Residual
    plt.plot(predicted, residual, '.', c='g')

    plt.plot(y, predicted, '.', c='b')

    plt.text(2, 50, 'Linearity Present')
    plt.xlabel('Independent Variable(x)')
    plt.ylabel('Dependent Variable(y)')

if __name__ == '__main__':
    #linear_plot_residual()
    non_linear_plot()
    plt.show()
