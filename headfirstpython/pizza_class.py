#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>


class Pizza:
    def __init__(self, ingredients):
        self.ingredients = ingredients

    def __repr__(self):
        return f'Pizza({self.ingredients!r})'

    @classmethod
    def class_method(cls) -> None:
        print(f'Calling Class Method with {cls}')

    @staticmethod
    def static_method() -> None:
        print(f'Calling Static method')

    @classmethod
    def margarita(cls):
        return cls(['mozzarella', 'tomato'])

    @classmethod
    def prosciutto(cls):
        return cls(['mozzarella', 'ham'])


if __name__ == '__main__':
    pizza = Pizza(['mozarella', 'tomato'])
    print(pizza.__class__.__class__)

    print(f'Has Attr {hasattr(Pizza, "random_func")}')

    def random_func() -> None:
        print("Random Func Called")

    Pizza.random_func = random_func

    print(f'Has Attr {hasattr(Pizza, "random_func")}')

    Pizza.random_func()

    # Usage of class method
    margarita = Pizza.margarita()
    prosciutto = Pizza.prosciutto()

    # Usage of Static Method
    # Used as a util method which does not use attributes of class
