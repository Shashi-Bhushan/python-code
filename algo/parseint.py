#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1@outlook.com>

def parseInt(str):
    print(f"String to convert to integer {str}")

    base = 1
    sum = 0

    for num in list(str):
        n = int(num) - int('0')
        sum += n * base
        base *= 10

    return sum

# Lookup table Approach

if __name__ == '__main__':
    print(type(parseInt('101')))
