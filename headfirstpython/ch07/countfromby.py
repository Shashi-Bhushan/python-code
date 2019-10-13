#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>


class CountFromBy:
    def __init__(self, val: int = 10, incr: int = 1) -> None:
        """Initializer for Count From By, takes value and increment"""
        self.val = val
        self.incr = incr

    def increment(self):
        """Increment the integer value by incr"""
        self.val += self.incr


if __name__ == '__main__':
    a = CountFromBy(10)
    print(a)
    # a.increment()
    CountFromBy.increment()
    print(a.val)
    print(id(a))
    print(hex(id(a)))
    print(hash(a))
