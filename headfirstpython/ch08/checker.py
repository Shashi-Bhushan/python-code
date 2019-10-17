#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>

from flask import session

from functools import wraps


def check_logged_in(func: object) -> object:
    """Decorator to allow access to only logged in user"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function that checks for session variable"""
        # 1. Code to execute BEFORE calling the decorated function (session variable check in this case)

        # 2. Call the decorated function as required, returning it's results if needed
        if 'logged_in' in session:
            return func(*args, *kwargs)
        else:
            return 'You are not logged in'

        # 3. Code to execute INSTEAD of calling the decorated function (Not logged in message in this case)

    return wrapper
