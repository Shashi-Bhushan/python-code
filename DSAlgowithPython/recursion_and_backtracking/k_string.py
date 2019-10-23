#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>

"""
Problem Statement: Generate all the strings of length n drawn from [0 .. k - 1]
"""


def base_k_string(n, k):
    if n == 0:
        return []
    elif n == 1:
        return [str(i) for i in range(k)]
    else:
        # Same principle as binary_strings,
        # Make a call to base case, here base_k_string(1, k) will give me a list of 1 digit numbers
        # Then, I will append each number from base case to other lists

        return [digit + bitstring for digit in base_k_string(1, k) for bitstring in base_k_string(n - 1, k)]


if __name__ == '__main__':
    print(base_k_string(2, 5))