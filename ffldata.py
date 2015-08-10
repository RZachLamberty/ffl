#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: ffldata.py
Author: zlamberty
Created: 2015-08-10

Description:
    wrapper for interacting with postgres db of web-scraped ffl data

Usage:
    <usage>

"""

import logging as logging
import pandas as pd
import sqlalchemy


# ----------------------------- #
#   Module Constants            #
# ----------------------------- #

logger = logging.getLogger(__name__)


# ----------------------------- #
#   Main routine                #
# ----------------------------- #

def load_prediction_data(source=None):
    """ load all the values where source == source (or no filtering, if source
        is None)

    """
    con = sqlalchemy.create_engine(
        "postgresql://ffldata:ffldata@localhost/ffldata"
    )
    if source:
        qry = "SELECT * FROM raw_data WHERE ffl_source = %(source)s ;"
        params = {'source': source}
    else:
        qry = "SELECT * FROM raw_data;"
        params = None
    d = pd.read_sql(sql=qry, con=con, params=params)

    # change WRRB into plain old WR
    d.loc[d.pos == 'WRRB', 'pos'] = 'WR'

    # make pos into a category
    d.loc[:, 'pos'] = d.loc[:, 'pos'].astype('category')

    return d
