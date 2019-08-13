#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1@outlook.com>

import pandas as pd

import util


def main():
    dataset = pd.read_csv('dataset/quarter.csv', sep=',')

    # Extract Dataset
    dataset1 = dataset.loc[:, ['x_1', 'y_1']]
    dataset2 = dataset.loc[:, ['x_2', 'y_2']]
    dataset3 = dataset.loc[:, ['x_3', 'y_3']]
    dataset4 = dataset.loc[:, ['x_4', 'y_4']]

    # Rename Columns
    dataset1.columns = ['x', 'y']
    dataset2.columns = ['x', 'y']
    dataset3.columns = ['x', 'y']
    dataset4.columns = ['x', 'y']

    util.plot_df(dataset1, 1)
    util.plot_df(dataset2, 2)
    util.plot_df(dataset3, 3)
    util.plot_df(dataset4, 4)


if __name__ == '__main__':
    main()
