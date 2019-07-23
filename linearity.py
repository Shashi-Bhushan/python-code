import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit

# Fixing random state for reproducibility
np.random.seed(100)

SIZE = 100


def get_linear_points():
    # X is range 0 to 50
    x = np.arange(0, SIZE, 2)
    # Y is X +/- a random number
    y = x + (np.random.rand(x.shape[0]) * 10)

    return x, y


def get_non_linear_points():
    # Because of a cubic function, x and y no longer share a linear relation

    # X is range 0 to 50
    x = np.arange(0, SIZE, 2)
    # Y is X^3 +/- a random number
    y = SIZE - (10 ** (x/49))

    return x, y


def plot(fig_index, independent_variable, marks_obtained, independent_variable_name):

    intercept, slope = polyfit(independent_variable, marks_obtained, 1)

    marks_predicted = intercept + slope * independent_variable

    plt.figure(fig_index)

    # Plot Observed(Actual) points
    plt.plot(independent_variable, marks_obtained, '.', label="Observed Points")

    # Plot Predicted points
    plt.plot(independent_variable, marks_predicted, '-', label="Predicted Points")

    plt.legend(loc="upper left", frameon=False)

    plt.xlabel(independent_variable_name)
    plt.ylabel('Marks Obtained')
    plt.title('Independent and Dependent Variable Plot')


def plot_observed_vs_predicted(fig_index, independent_variable, marks_obtained, independent_variable_name):
    intercept, slope = polyfit(independent_variable, marks_obtained, 1)

    marks_predicted = (intercept + slope * independent_variable)

    plt.figure(fig_index)

    plt.plot(marks_obtained, marks_predicted, '.', c='g')

    plt.xlabel('Observed Values for Marks Obtained')
    plt.ylabel('Predicted Values for Marks Obtained')
    plt.title('Observed vs Predicted Plot')


def plot_predicted_vs_residual(fig_index, independent_variable, marks_obtained, independent_variable_name):
    intercept, slope = polyfit(independent_variable, marks_obtained, 1)

    marks_predicted = (intercept + slope * independent_variable)

    residual = marks_obtained - marks_predicted

    plt.figure(fig_index)

    # Plot Predicted vs Residual Graph
    plt.plot(marks_predicted, residual, '.', c='g')
    plt.vlines(marks_predicted, 0, residual, linestyles='dashed')

    plt.xlabel('Predicted Values for Marks Obtained')
    plt.ylabel('Residual')
    plt.title('Predicted vs Residual Plot')


if __name__ == '__main__':
    plot(1, get_linear_points()[0], get_linear_points()[1], "Number of Hours Studied")
    plot_observed_vs_predicted(2, get_linear_points()[0], get_linear_points()[1], "Number of Hours Studied")
    plot_predicted_vs_residual(3, get_linear_points()[0], get_linear_points()[1], "Number of Hours Studied")

    plot(4, get_non_linear_points()[0], get_non_linear_points()[1], "Number of Hours Played")
    plot_observed_vs_predicted(5, get_non_linear_points()[0], get_non_linear_points()[1], "Number of Hours Played")
    plot_predicted_vs_residual(6, get_non_linear_points()[0], get_non_linear_points()[1], "Number of Hours Played")
    plt.show()
