#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


def show_plots(X):
    plt.figure(1)
    sns.pairplot(X, height=4, aspect=1, kind='scatter')
    plt.show()

    plt.figure(2)
    sns.heatmap(X.corr(), cmap='YlGnBu', annot=True)
    plt.show()


def main():
    iris_dataset = datasets.load_iris()
    X = pd.DataFrame(iris_dataset.data)
    X.columns = iris_dataset.feature_names
    y = pd.DataFrame(iris_dataset.target, columns=['flower type'])

    #show_plots(X)

    X.drop(['sepal length (cm)', 'petal length (cm)'], axis=1, inplace=True)

    # Split to Test and Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=105)

    # StatsModel
    X_train_sm = sm.add_constant(X_train)
    sm_model = sm.OLS(y_train, X_train_sm).fit()

    print(sm_model.summary())

    vif = pd.DataFrame()
    vif['Features'] = X_train.columns

    vif['VIF'] = [variance_inflation_factor(X_train.values, index) for index in range(X_train.shape[1])]

    print(vif)


if __name__ == '__main__':
    main()

