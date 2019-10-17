#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>

from flask import session

from functools import wraps


def check_logged_in(func):
    """Decorator to allow access to only logged in user"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function that checks for session variable"""

        if 'logged_in' in session:
            return func(*args, *kwargs)
        else:
            return 'You are not logged in'

    return wrapper
