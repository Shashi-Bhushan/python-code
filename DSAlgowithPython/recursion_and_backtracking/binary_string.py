#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>

"""
Problem Statement: Generate all the binary strings with n bits.
"""

from itertools import product


def binary_string_non_recur(num: int) -> list:
    """Generate all binary strings with n bits Non recursively"""
    return [''.join(p) for p in product('10', repeat=num)]


def binary_strings_for_loop(num: int) -> list:
    """Generate all binary strings with n bits recursively"""
    if num == 0:
        return []
    elif num == 1:
        return ['0', '1']
    else:
        # In base condition, binary_strings will return ['0', '1']
        # for num = 2, I will append 0 to this making it ['00', '01']
        # Similarly, I will also append 1 to this making it ['10', '01']
        # In the end, I will add these two lists together. In python, + does that for me.

        # return ['0' + digit for digit in binary_strings(num - 1)] + ['1' + digit for digit in binary_strings(num - 1)]

        # Since I could fetch ['0', '1'] with binary_string(1) call,
        # lets replace this with another for loop just to make the solution generic enough

        return [bit + digit for bit in binary_strings_for_loop(1) for digit in binary_strings_for_loop(num - 1)]


def binary_string(depth: int):
    """
    Returns Binary String having specified depth
    :type depth: int
    """
    if depth == 0:
        return []
    elif depth == 1:
        return ['0', '1']
    else:
        l1 = binary_string(depth - 1)
        l2 = binary_string(depth - 1)

        for index, item in enumerate(l1):
            l1[index] = '0' + l1[index]
            l2[index] = '1' + l2[index]

        return l1 + l2


if __name__ == '__main__':
    n = 3
    bin_non_recur = binary_string_non_recur(n)

    print(f'Binary Strings with {n} bits : {bin_non_recur}')

    bin_recur = binary_strings_for_loop(n)

    print(f'Binary Strings (using Recursion) with {n} bits : {bin_recur}')

