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


def binary_strings(num: int) -> list:
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

        return ['0' + digit for digit in binary_strings(num - 1)] + ['1' + digit for digit in binary_strings(num - 1)]


if __name__ == '__main__':
    n = 3
    bin_non_recur = binary_string_non_recur(n)

    print(f'Binary Strings with {n} bits : {bin_non_recur}')

    bin_recur = binary_strings(n)

    print(f'Binary Strings (using Recursion) with {n} bits : {bin_recur}')

