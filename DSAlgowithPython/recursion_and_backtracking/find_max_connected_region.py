#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>

"""
Problem Statement: Finding the length of connected cells of 1s (regions) in a matrix of 0s and 1s.
"""

from pprint import pprint

# The simplest idea is for each location, traverse in all 8 directions.
# In each of those directions, keep track of maximum region found.


def get_val(arr: object, row: int, column: int, row_max: int, column_max: int) -> int:
    """Get value of the item at position (row, column) iff it's within bounds of the matrix"""
    if row < 0 or row_max <= row or column < 0 or column_max <= column:
        return 0
    else:
        return arr[row][column]


def find_max_block(arr: object, row: int, column: int, row_max: int, col_max: int, size: int) -> int:
    """Recursive function to calculate size"""

    if row_max <= row or col_max <= column:
        return size

    size = size + 1

    # Search in all 8 directions
    directions = ([-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1])

    for index in range(len(directions)):
        new_row = row + directions[index][0]
        new_column = column + directions[index][1]

        val = get_val(arr, new_row, new_column, row_max, col_max)

        if 0 < val:
            return find_max_block(arr, new_row, new_column, row_max, col_max, size)


def get_num_max_ones(arr: object, row_max: int, col_max: int) -> int:
    max_block = 0

    for row in range(row_max):
        for column in range(col_max):
            if arr[row][column] == 1:
                local_max_block = find_max_block(arr, row, column, row_max, col_max, 0)

                if max_block < local_max_block:
                    max_block = local_max_block

    return max_block


if __name__ == '__main__':
    arr = [
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 1]
    ]

    max_ones = get_num_max_ones(arr, 5, 5)

    print('Array is : ')
    pprint(arr)

    print(f'Max Region val is : {max_ones}')
