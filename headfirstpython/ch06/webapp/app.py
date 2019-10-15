#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>

from flask import Flask, request, render_template, escape

from vsearch import search_for_letters

from dbcm import UseDatabase

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
    """Display the contents of the log file as a String table."""
    contents = []

    with open(LOG_FILE) as app_log:
        for line in app_log:
            contents.append([])

            for item in line.split('|'):
                contents[-1].append(escape(item))

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
