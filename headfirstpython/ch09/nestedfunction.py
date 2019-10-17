#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>


def outer(index: int) -> object:
    private_var = "This is private string var"

    def inner(index: int) -> str:
        return private_var + ' with value ' + str(index)

    return inner(index)


def var_arg_func(*args: tuple) -> None:
    """args is a tuple of values"""

    print('Printing Var args')
    for arg in args:
        print(arg, end=' ')

    print()
    print('Finished Printing Var args')


if __name__ == '__main__':
    string_val = outer(5)
    print(string_val)

    var_arg_func()
    var_arg_func(1, 3)
    var_arg_func(1)
