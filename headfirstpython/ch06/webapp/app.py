#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>

from flask import Flask, request, render_template, escape

from vsearch import search_for_letters

from dbcm import UseDatabase, DBConnectionError, CredentialsError, SQLError

app = Flask(__name__)
app.config['dbconfig'] = {
    'host': '127.0.0.1',
    'user': 'vsearch',
    'password': 'vsearchpasswd',
    'database': 'vsearchlogDB',
}
LOG_FILE = 'app.log'


@app.route('/')
@app.route('/entry')
def home_page() -> 'html':
    """Display this webapp's HTML form."""
    return render_template('searchform.html', the_title='Welcome to Search for Letters web app')


@app.route('/search4', methods=['POST'])
def do_search() -> 'html':
    """Extract the posted data, perform the search and return results."""
    phrase = request.form['phrase']
    letters = request.form['letters']

    results = str(search_for_letters(phrase, letters))
    log_request(request, results)

    return render_template('searchresults.html',
                           the_title='Search Results',
                           the_phrase=phrase,
                           the_letters=letters,
                           the_results=results
                           )


@app.route('/viewlog')
def view_log() -> str:
    """Display the contents of the log file as a String table.

    Don't import mysql errors here, since it will tightly couple your code with MySQL code.
    All the code specific to Database should be inside dbcm.py only.
    """
    try:
        with UseDatabase(app.config['dbconfig']) as cursor:
            _SQL = """select phrase, letters, ip, browser_string, results
                      from log"""
            cursor.execute(_SQL)
            contents = cursor.fetchall()
        titles = ('Phrase', 'Letters', 'Remote Address', 'User agent', 'Results')

        return render_template('viewlog.html',
                               the_title='View Log',
                               the_row_titles=titles,
                               the_data=contents, )
    except DBConnectionError as err:
        print('Is your database switched on? Error:', str(err))
    except CredentialsError as err:
        print('User-id/Password issues. Error:', str(err))
    except SQLError as err:
        print('Is your query correct? Error:', str(err))
    except Exception as err:
        print('Something went wrong:', str(err))
    return 'Error'


def log_request(req: 'flask_request', res: str) -> None:
    """Log details of the web request and the results."""
    with open(LOG_FILE, 'a') as app_log:
        print(req.form, req.remote_addr, req.user_agent, req.form['phrase'], req.form['letters'], res, file=app_log,
              sep='|')

    with UseDatabase(app.config['dbconfig']) as cursor:
        _SQL = """insert into log
                          (phrase, letters, ip, browser_string, results)
                          values
                          (%s, %s, %s, %s, %s)"""
        cursor.execute(_SQL, (req.form['phrase'],
                              req.form['letters'],
                              req.remote_addr,
                              req.user_agent.browser,
                              res,))


if __name__ == '__main__':
    app.run()
