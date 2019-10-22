#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>

"""
Problem Statement: Check if the array is sorted by using Recursion
"""


def check_sorted(arr: list) -> bool:
    # arr[0] = 0

    if len(arr) == 1:
        return True
    else:
        print(f'Inside Function, Map for {arr} : {list(map(id, arr))}')
        return arr[0] <= arr[1] and check_sorted(arr[1:])


if __name__ == '__main__':
    arr = [1, 2, 4, 5, 6]

    # When I'm doing a function call, the function will receive a reference to the same Array object arr.
    # That's why, inside function when I change arr[0] = 0, the value in the original array object is also updated.
    # It's a plain call by reference.
    #
    # when I slice the array object, a new array object is created copying the same references as original array.
    # We could confirm this by calling check_sorted(arr[:])
    # and confirming that inside function, arr[0] = 0 does not change the original array in this case.
    print(f'Outside Function, Map : {list(map(id, arr))}')
    is_sorted = check_sorted(arr)
    print(f'Map : {list(map(id, arr))}')

    print(f'\nArray {arr} is sorted in ascending order : {is_sorted}')
