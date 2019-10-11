#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>

from flask import Flask, request, render_template

from vsearch import search_for_letters

app = Flask(__name__)


@app.route('/')
@app.route('/entry')
def home_page() -> 'html':
    return render_template('searchform.html', the_title='Welcome to Search for Letters web app')


@app.route('/search4', methods=['POST'])
def do_search() -> 'html':
    phrase = request.form['phrase']
    letters = request.form['letters']

    return render_template('searchresults.html',
                           the_title='Search Results',
                           the_phrase=phrase,
                           the_letters=letters,
                           the_results=str(search_for_letters(phrase, letters))
                           )


app.run()
