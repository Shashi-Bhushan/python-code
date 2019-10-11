#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>


import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit


def plot_df(dataframe, fignum, title):
    plt.figure(fignum, figsize=(6, 6), dpi=60)

    plt.plot(dataframe['x'], dataframe['y'], 'o', c='g')

    intercept, slope = polyfit(dataframe['x'], dataframe['y'], 1)

    regression_line = intercept + slope * dataframe['x']

    plt.plot(dataframe['x'], regression_line, '-', color='orange')

    plt.xlim(3, 20)
    plt.ylim(3, 13)

    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")

    plt.title(title)

    plt.show()
