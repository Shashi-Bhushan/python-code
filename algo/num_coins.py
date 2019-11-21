#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Shashi Bhushan (sbhushan1 @ outlook dot com)'


def num_coins_ver_1(cents):
    """33 cents = 1 quarter (25 cents), 1 dime (10 cents), 1 nickle (5 cents) and 3 cents

    This algorithm greedily counts how many coins of the highest denomination available will be needed.

    Problem:
    Suppose machine ran out of nickle (5 cents). Then, the algorithm will give 7 as number of coins
    7 = 1 quarter + 6 cents
    but the optimal solution is 3 dime and 1 cent (4 coins total). we'll solve this in next version.
    """

    coins = [25, 10, 5, 1]

    num_of_coins = 0
    for coin in coins:
        num_of_coins += cents // coin
        cents = cents % coin

    return num_of_coins


if __name__ == '__main__':
    print(num_coins_ver_1(31))
