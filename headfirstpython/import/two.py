#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>

import sys

x = input('Press Key')

while x != 'q':
    if x == 'f':
        import one
        one.test()
    elif x == 's':
        print('S is pressed\n')
        print(sys.modules)

    x = input('Press Key Again \n')

print("Quitting")
