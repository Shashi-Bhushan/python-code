#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>

from flask import Flask, session

from checker import check_logged_in

app = Flask(__name__)
app.secret_key = 'secret_key'


@app.route('/')
def hello() -> str:
    return 'Hello from the simple web app'


@app.route('/login')
def do_login() -> str:
    session['logged_in'] = True
    return 'You are now logged in'


@app.route('/logout')
def do_logout() -> str:
    session.pop('logged_in', None)
    return 'You are now logged out'


@app.route('/status')
def status() -> str:
    if 'logged_in' in session:
        return 'User is logged in'
    elif 'logged_in' not in session:
        return 'User not logged in'
    else:
        return 'Unidentified error'


@app.route('/page1')
@check_logged_in
def page1() -> str:
    return 'This is Page One.'


@app.route('/page2')
def page2() -> str:
    return 'This is Page Two.'


if __name__ == '__main__':
    app.run(debug=True)
