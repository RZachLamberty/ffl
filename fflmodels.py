#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: fflmodels.py
Author: zlamberty
Created: 2015-08-10

Description:
    modeling and statistics for ffl data sets

Usage:
    <usage>

"""

import logging
import pandas as pd

import ffldata

from sklearn.linear_model import LinearRegression


# ----------------------------- #
#   Module Constants            #
# ----------------------------- #

logger = logging.getLogger(__name__)


# ----------------------------- #
#   Main routine                #
# ----------------------------- #

def positional_hist(df, minthresh=10):
    x = df.loc[df.pts_total.ge(minthresh)]
    x.hist('pts_total', by='pos', bins=25)


def repl_val_hist(df):
    x = df.drop(df.groupby('pos').repl_val.idxmax())
    x = x.loc[x.pts_total.gt(0)]
    x.hist('pts_total', by='pos', bins=25)


def lm(df):
    features = df.columns[5: 16]

    scoring = pd.DataFrame(columns=features.tolist() + ['intercept', 'score'])
    for (pos, posdata) in df.groupby('pos'):
        if pos not in ['D/ST', 'K']:

            X = posdata[features]
            Y = posdata.pts_total

            lm = LinearRegression()
            lm.fit(X, Y)

            coefs = pd.Series(
                {cn: coef for (cn, coef) in zip(features, lm.coef_)},
                name=pos
            )
            coefs['intercept'] = lm.intercept_
            coefs['score'] = lm.score(X, Y)

            scoring = scoring.append(coefs)

    return scoring


def score_comparison(dftest, dftrial):
    models = {}
    features = dftest.columns[5: 16]
    scoring = pd.DataFrame(columns=features.tolist())
    for (pos, posdata) in dftest.groupby('pos'):
        if pos not in ['D/ST', 'K']:
            X = posdata[features]
            Y = posdata.pts_total

            lm = LinearRegression()
            lm.fit(X, Y)

            models[pos] = lm

    def f(chunk):
        pos = chunk.poscopy.unique()[0]
        if pos not in ['D/ST', 'K']:
            lm = models[pos]
            X = chunk[features]
            Y = lm.predict(X)
            return pd.DataFrame(Y, index=chunk.index, columns=['pred_pts_total'])
        else:
            return pd.Series(None, index=chunk.index, columns=['pred_pts_total'])

    dftrialcopy = dftrial.copy()
    dftrialcopy.loc[:, 'poscopy'] = dftrialcopy.loc[:, 'pos']

    gb = dftrialcopy.groupby('pos')
    dftrial.loc[:, 'pred_pts_total'] = gb.apply(f)

    return dftrial


def demo_score_comparison(testsource='espn', trialsource='cbs'):
    dftest = ffldata.load_prediction_data(source=testsource)
    dftrial = ffldata.load_prediction_data(source=trialsource)

    dftrialpred = score_comparison(dftest, dftrial)
    m = pd.merge(
        dftrialpred, dftest, how='left', on='playername',
        suffixes=('_{}'.format(src) for src in [testsource, trialsource])
    )

    return m
