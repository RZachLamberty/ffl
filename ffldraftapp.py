#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: ffldraftapp.py
Author: zlamberty
Created: 2016-09-10

Description:
    flask app for interactive drafting

Usage:
    <usage>

"""

from flask import Flask, g, json, render_template, request
from flask_bootstrap import Bootstrap

from ffldraft import DraftData
from forms import DraftForm, update_draftform_choices


app = Flask(__name__)
app.config.from_object('config')
FFLDRAFT = None

Bootstrap(app)


@app.route('/', methods=('GET', 'POST'))
def index():
    # make sure we have created draft data before
    global FFLDRAFT
    if FFLDRAFT is None:
        print('building ffldraft object')
        FFLDRAFT = DraftData(app.config['FYAML'])
    g.ffldraft = FFLDRAFT

    # if we posted a draft, make it happen
    if request.method  == 'POST':
        g.ffldraft.draft_by_ids(
            pid=int(request.form['playername']),
            tid=int(request.form['draftteam']),
        )

    replacementPlotData = json.dumps(
        g.ffldraft.plotly_best_replacement_available(
            app.config['REPL_PLOT_NUM']
        )
    )

    # form for declaring a player drafted
    draftform = DraftForm()
    update_draftform_choices(draftform, g.ffldraft)

    return render_template(
        'index.html',
        replacementPlotData=replacementPlotData,
        draftform=draftform
    )


@app.route('/undo_draft', methods=('POST',))
def undo_draft():
    global FFLDRAFT
    FFLDRAFT


if __name__ == '__main__':
    app.run(debug=True)
