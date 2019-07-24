import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd

# Fixing random state for reproducibility
np.random.seed(100)

SIZE = 100

def get_linear_points():
    """
        Returns a Linear Dataset with x and y

        It has been sourced from https://www.kaggle.com/andonians/random-linear-regression#train.csv
    :return:
        a linear dataset
    """
    df = pd.read_csv('dataset/linear.csv')

    return df['x'], df['y']


def get_non_linear_points():
    # Because of a cubic function, x and y no longer share a linear relation

    # X is range 0 to 50
    x = np.arange(0, SIZE, 2)
    # Y is X^3 +/- a random number
    y = SIZE - (10 ** (x/49))

    return x, y


def plot(fig_index, get_points, independent_variable_name):
    independent_variable, marks_obtained = get_points

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


def plot_observed_vs_predicted(fig_index, get_points, independent_variable_name):
    independent_variable, marks_obtained = get_points

    intercept, slope = polyfit(independent_variable, marks_obtained, 1)

    marks_predicted = (intercept + slope * independent_variable)

    plt.figure(fig_index)

    plt.plot(marks_obtained, marks_predicted, '.', c='g')

    plt.xlabel('Observed Values for Marks Obtained')
    plt.ylabel('Predicted Values for Marks Obtained')
    plt.title('Observed vs Predicted Plot')


def plot_predicted_vs_residual(fig_index, get_points, independent_variable_name):
    independent_variable, marks_obtained = get_points

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


def plot_residual_frequency_dist(fig_index, get_points, independent_variable_name):
    independent_variable, marks_obtained = get_points
    intercept, slope = polyfit(independent_variable, marks_obtained, 1)

    marks_predicted = (intercept + slope * independent_variable)

    residual = marks_obtained - marks_predicted

    plt.figure(fig_index)

    # Plot Predicted vs Residual Graph
    plt.hist(residual, bins=np.arange(-5.0, 5.0, 0.5), histtype='step')
    y, binEdges = np.histogram(residual, bins=np.arange(-5.0, 5.0, 0.5))
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    plt.plot(bincenters, y, '-')

    #import probscale
    #probscale.probplot(residual, problabel='Standard Normal Quantiles')

    plt.xlabel('Error Terms')
    plt.ylabel('Frequency')
    plt.title('Residual Frequency Plot')

if __name__ == '__main__':
    linear_points = get_linear_points()
    #plot(1, linear_points, "Number of Hours Studied")
    #plot_observed_vs_predicted(2, linear_points, "Number of Hours Studied")
    #plot_predicted_vs_residual(3, linear_points, "Number of Hours Studied")
    #plot_residual_frequency_dist(4, linear_points, "Number of Hours Studied")

    non_linear_points = get_non_linear_points()

    plot(5, non_linear_points, "Number of Hours Played")
    #plot_observed_vs_predicted(6, non_linear_points, "Number of Hours Played")
    #plot_predicted_vs_residual(7, non_linear_points, "Number of Hours Played")
    plot_residual_frequency_dist(8, non_linear_points, "Number of Hours Played")
    plt.show()
