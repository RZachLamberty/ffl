#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
module: ffldraft.py
author: Zach Lamberty
created: 2013-08-30

Description:
    <desc>

Usage:
    <usage>

"""

import csv
import datetime
import itertools
import logging
import os

import pandas as pd
import pylab
import seaborn as sns
import yaml

import ffldata

from collections import defaultdict
from math import ceil, floor

pylab.close('All')


#---------------------------#
#   Module Constants        #
#---------------------------#

HERE = os.path.realpath(os.path.dirname(__file__))
SASNUT = os.path.join(HERE, 'teams.sasnut.yaml')
LAMBO = os.path.join(HERE, 'teams.lambo.yaml')
DEMO = os.path.join(HERE, 'teams.demo.yaml')

# logging
logger = logging.getLogger(__name__)


#---------------------------#
#   Live Draft Class        #
#---------------------------#

class DraftData():
    """ A class object to calculate draft data, who to pick, etc. """
    def __init__(self, teamYaml=SASNUT):
        # loading team info
        self.teamYaml = teamYaml
        self._teamdict = None
        self._leagueteams = None

        self.drafthistory = []
        self.data = ffldata.load_prediction_data(source='espn')
        self.update_replacement_value()

        #self.f1 = pylab.figure(1, figsize=[7.5, 5.5])
        #self.f2 = pylab.figure(2, figsize=[12.5, 7.5])
        #self.fScratch = pylab.figure(3, figsize=[7.5, 5.5])

        #self.show_best_replacement_available()

    # loading team info --------------------------------------------------------
    @property
    def teamdict(self):
        if self._teamdict is None:
            with open(self.teamYaml, 'r') as f:
                self._teamdict = yaml.load(f)
        return self._teamdict

    @property
    def leagueteams(self):
        if self._leagueteams is None:
            self._leagueteams = pd.DataFrame(self.teamdict['teams'])
        return self._leagueteams

    @property
    def teamlist(self):
        return sorted(self.leagueteams.code.unique())

    @property
    def positions(self):
        return self.teamdict['positions']

    # Updating who has been drafted -- interactive -----------------------------
    @property
    def undrafted(self):
        return self.data[self.data.status_type == 'FA']

    def get_player_interactive(self, allowDrafted=False):
        """ Allow the user to select a player by initials """
        # Prompt the user to choose who has been drafted and on which team
        first = input('First part of first name?\t').lower()
        last = input('First part of last name? \t').lower()

        print('\n')

        names = self.data.playername.str.lower()
        matches = self.data[names.str.match(r'^{}.* {}.*'.format(first, last))]
        matches = matches[(matches.status_type == 'FA') | (allowDrafted)]

        print(matches[['playername', 'team']])

        ind = int(input('which index is it? '))

        try:
            return matches.loc[ind]
        except KeyError:
            logger.error("that is an invalid index! try again!")
            return None

    def get_team_interactive(self):
        """ Allow the user to select a team by team_id """
        print('\n')
        print(self.leagueteams)

        ind = int(input('which index do you want? '))

        try:
            return self.leagueteams.loc[ind]
        except KeyError:
            logger.error("that is an invalid index! try again!")
            return None

    def new_draft_interactive(self):
        """ Take in a draft notice and update everything accordingly """
        player = self.get_player_interactive()

        if player is not None:
            team = self.get_team_interactive()
            self.been_drafted(player, team)

    def false_draft_interactive(self):
        """ Undo a previous draft that has been rolled back (it always happens) """
        player = self.get_player_interactive(allowDrafted=True)
        self.been_drafted(player, None)

    def been_drafted(self, player, team, updateplots=True):
        """ Update the prediction data to indicate a draft """
        if team is not None:
            self.data.loc[player.name, 'status_type'] = team.code
            logger.info(
                '{} has been drafted by {}'.format(player.playername, team.code)
            )
            self.drafthistory.append({'player': player.name, 'team': team.name})
        else:
            self.data.loc[player.name, 'status_type'] = 'FA'
            logger.info('{} has been undrafted'.format(player.playername))
            self.drafthistory = [
                d for d in self.drafthistory
                if not d['player'] == player.name
            ]

        self.update_replacement_value()
        if updateplots:
            self.show_best_replacement_available()
            #self.state_of_draft(poslist)

    # Replacement Value Calculation --------------------------------------------
    def update_replacement_value(self):
        """ For every undrafted player, calculate their replacement value at
            their position.  We may generalize this

        """
        # make sure it's sorted
        self.data.sort_values(
            by=['pos', 'pts_total', 'status_type'],
            inplace=True
        )

        # now that it's sorted, series.diff gives us repl vals
        g = self.data.groupby('pos')['pts_total']
        self.data.loc[:, 'repl_val'] = g.transform(pd.Series.diff)

        # invert for calculating cumulative stats
        self.data.sort_values(
            by=['pos', 'pts_total', 'status_type'],
            ascending=[True, False, True],
            inplace=True
        )

        # add the cumulative stats
        undrafted = self.data[self.data.status_type == 'FA']
        g = undrafted[['pos', 'repl_val']].groupby('pos')
        crv = g.cumsum()
        crv.columns = ['cum_repl_val']
        ct = g.cumcount()
        ct.columns = ['cum_count']
        self.data.loc[self.data.status_type == 'FA', 'cum_repl_val'] = crv
        self.data.loc[self.data.status_type == 'FA', 'cum_count'] = ct

    def show_best_replacement_available(self, N=25):
        """ Extract the best N remaining players at each position, and determine
            what their replacement values are. Plot the series of replacement
            picks to give a future-looking view on the decision to not draft the
            calculated top remaining player

        """
        logger.info("calculating the top {} available at each position".format(N))

        # best available
        f = pylab.figure(figsize=[7.5, 5.5])
        s1 = f.add_subplot(111)
        g = self.data[self.data.status_type == 'FA'].groupby('pos')
        for (pos, group) in g:
            lab = '{}: {}'.format(pos, group.iloc[0].playername)
            group.head(25).plot(
                x='cum_count', y='cum_repl_val', style='o-', ax=s1, label=lab
            )

        s1.set_xlim((-0.5, 25.5))

        logger.debug("Displaying")
        return f

    def plotly_best_replacement_available(self, N=25):
        """ create a json-able object that can be plotted with plotly.js """
        g = self.data[self.data.status_type == 'FA'].groupby('pos')
        return [
            {
                'x': group.head(N).cum_count.values.tolist(),
                'y': group.head(N).cum_repl_val.values.tolist(),
                'text': group.head(N).playername.values.tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': '{}: {}'.format(pos, group.iloc[0].playername)
            }
            for (pos, group) in g
        ]

    def state_of_draft(self):
        """ Create an N-team panelled histogram plot which shows how each team
            is faring relative to the median value at each position, as well as
            how that would change if the best X is chosen in this next draft.
            To be updated and displayed after every draft pick.

            Perhaps also include a "residual"-esque plot of the pts lost by not
            chosing the next best player at that position (i.e. points lost to
            present mean, not by forgoing for replacement)

        """
        positions = self.data.pos.unique()

        # list of all teams who have made draft picks so far
        teamSummary = self.team_draft_summary(self.positions)
        starters = teamSummary.loc[teamSummary.starting_pos.notnull()]

        # Total
        total = starters.groupby('status_type').pts_total.sum()
        totmedian = total.median()
        totmean = total.mean()
        totdelta = total - totmean

        # for (team, teamVals) in teamSummary.items():
        #     teamVals['TOTAL'] = {'PTS': sum(v.get('PTS', 0)
        #                                     for (k, v) in teamVals.items()
        #                                     if not 'flex_' in k)}

        # Calculate the median value at each position
        sgb = starters.groupby('starting_pos')
        median = sgb.pts_total.median()
        mean = sgb.pts_total.mean()

        # pos-by-pos delta values. The value of a player - the median value
        # of current starters
        starters.loc[:, 'pts_delta'] = starters.apply(
            func=lambda x: x.pts_total - median[x.starting_pos],
            axis=1
        )

        # Plot that shit
        N = len(starters.status_type.unique())
        J = floor(N ** .5)
        I = ceil(N / float(J))

        f = pylab.figure(figsize=[12.5, 7.5])
        f.subplots_adjust(
            left=0.04, bottom=0.06, right=0.98, top=0.95, hspace=1.0
        )

        poslistSpec = self.positions + ['TOTAL']
        lefts = [i for i in range(len(poslistSpec))]
        width = 0.4
        mids = [i + width for i in range(len(poslistSpec))]
        maxdelta = starters.pts_delta.max()
        mindelta = starters.pts_delta.min()

        sgb = starters.groupby('status_type')
        for (i, (team, teamgrp)) in enumerate(sgb):
            sNow = f.add_subplot(I, J, i + 1)

            # collect starters and total values for plotting
            startplt = teamgrp[['starting_pos', 'pts_delta']]
            totdf = pd.DataFrame(
                data=[{'starting_pos': 'TOTAL', 'pts_delta': totdelta[team]}],
                columns=['starting_pos', 'pts_delta']
            )
            combo = pd.concat([teamgrp[['starting_pos', 'pts_delta']], totdf])
            colors = ['blue'] * combo.shape[0]
            colors[-1] = 'green' if totdf.iloc[0].pts_delta > 0 else 'red'

            # current vals -- starters
            combo.plot(
                x='starting_pos', y='pts_delta', kind='bar', ax=sNow,
                alpha=0.75, color=colors, legend=False, rot=70
            )

            sNow.set_title(team)
            sNow.set_xlabel('')
            sNow.set_ylim((mindelta, maxdelta))

        logger.debug("Displaying")
        return f

    def team_draft_summary(self):
        """ Return a list of all drafted players, binned by positions (best
            available)

        """
        tds = {}
        # collect all drafted players
        drafted = self.data[self.data.status_type != 'FA']

        for (team, roster) in drafted.groupby('status_type'):
            tds[team] = self.best_lineup(roster, self.positions)

        return pd.concat(tds.values())

    def best_lineup(self, roster, poslist):
        roster.loc[:, 'starting_pos'] = None

        multiPos = [pos for pos in poslist if len(pos) > 1]
        overlapPos = {p for pos in multiPos for p in pos}
        singlePos = [pos for pos in poslist if len(pos) == 1]

        # just get the best player in starting pos where there is no positional
        # flexibility
        for pos in singlePos:
            try:
                inpos = roster[
                    (roster.pos == pos[0]) & (roster.starting_pos.isnull())
                ]
                bestind = inpos.pts_total.idxmax()
                roster.loc[bestind, 'starting_pos'] = pos
            except:
                continue

        # now, determine which combination of remaining players gives us the
        # best starting lineup (pts-wise)
        nonstarters = roster[
            roster.starting_pos.isnull() & roster.pos.isin(overlapPos)
        ]
        bestscore = 0
        bestiset = bestposset = None
        if not nonstarters.empty:
            inds = nonstarters.index
            N = min(len(inds), len(multiPos))

            # one ordered list of indices
            for iset in itertools.combinations(inds, N):
                # all N-length position settings
                for posset in itertools.permutations(multiPos, N):
                    posmatch = all([
                        nonstarters.loc[i, 'pos'] in pos
                        for (i, pos) in zip(iset, posset)
                    ])
                    if posmatch:
                        score = nonstarters.loc[iset, 'pts_total'].sum()
                        if score > bestscore:
                            bestscore = score
                            bestiset = iset
                            bestposset = posset
                        # we won't do better for this iset, so break
                        break

        # update the roster with new starters if they exist
        if bestiset is not None:
            roster.loc[bestiset, 'starting_pos'] = pd.Series(
                data=bestposset, index=bestiset, name='starting_pos'
            )

        return roster

    def best_lineup_hardcoded(self, roster):
        """ forget the flexibiliy; assume the position list """
        roster.loc[:, 'starting_pos'] = None

        # best single qb, wr, etc
        for pos in ['QB', 'WR', 'RB', 'TE', 'D/ST', 'K']:
            try:
                roster.loc[
                    roster.index[roster.pos == pos][0],
                    'starting_pos'
                ] = pos
            except:
                continue

        # best rb, wr, or te:
        bestRemainingRwt = roster.loc[
            roster.index[
                (
                    (roster.pos == 'RB')
                    | (roster.pos == 'WR')
                    | (roster.pos == 'TE')
                )
                & (roster.starting_pos.isnull())
            ]
        ].head(3)

        raise NotImplementedError("you haven't finished this funciton")

        multiPos = [pos for pos in poslist if len(pos) > 1]
        singlePos = [pos for pos in poslist if len(pos) == 1]

        # just get the best player in starting pos where there is no positional
        # flexibility
        for pos in singlePos:
            try:
                inpos = roster[
                    (roster.pos == pos[0]) & (roster.starting_pos.isnull())
                ]
                bestind = inpos.pts_total.idxmax()
                roster.loc[bestind, 'starting_pos'] = pos
            except:
                continue

        # now, determine which combination of remaining players gives us the
        # best starting lineup (pts-wise)
        nonstarters = roster[roster.starting_pos.isnull()]
        bestscore = 0
        bestiset = bestposset = None
        if not nonstarters.empty:
            inds = nonstarters.index
            N = min(len(inds), len(multiPos))

            # one ordered list of indices
            for iset in itertools.combinations(inds, N):
                # all N-length position settings
                for posset in itertools.permutations(multiPos, N):
                    posmatch = all([
                        nonstarters.loc[i, 'pos'] in pos
                        for (i, pos) in zip(iset, posset)
                    ])
                    if posmatch:
                        score = nonstarters.loc[iset, 'pts_total'].sum()
                        if score > bestscore:
                            bestscore = score
                            bestiset = iset
                            bestposset = posset
                        # we won't do better for this iset, so break
                        break

        # update the roster with new starters if they exist
        if bestiset is not None:
            roster.loc[bestiset, 'starting_pos'] = pd.Series(
                data=bestposset, index=bestiset, name='starting_pos'
            )

        return roster

    def state_of_position(self, N=25, pos='QB'):
        """ Same as above, but for only one position and with all points labeled
            by position name

        """
        topn = self.data[
            (self.data.status_type == 'FA') & (self.data.pos == pos)
        ].head(N)

        f = pylab.figure(figsize=[7.5, 5.5])
        s3 = f.add_subplot(111)

        try:
            best = topn.iloc[0]
            lab = "Top {} {}".format(N, pos)
            topn.plot(
                x='cum_count', y='cum_repl_val', style='o-', label=lab, ax=s3
            )

            # Annotating with player names
            for (index, row) in topn.iterrows():
                s3.annotate(
                    '{}, {}'.format(row.playername, row.team),
                    xy=(row.cum_count, row.cum_repl_val),
                    xytext=(row.cum_count, 0.975 * row.cum_repl_val),
                    rotation=-50,
                    horizontalalignment='left',
                    verticalalignment='top',
                )

            # move the plot down 20
            ylim = s3.get_ylim()
            s3.set_ylim((ylim[0] - 20.0, ylim[1]))

        except:
            logger.error('fuuuuuck')
            logger.info("pos  = {}".format(pos))
            logger.info("topN = {}".format(topN))
            raise

        s3.legend(loc='upper right', fontsize=10)

        logger.debug("Showing state of {}".format(pos))
        return f

    # draft history IO ---------------------------------------------------------
    @property
    def form_history(self):
        """a version of history which can more easily be passed to a web form"""
        return pd.DataFrame(
            [
                {
                    'draftnum': i,
                    'player_index': d['player'],
                    'player_name': self.data.loc[d['player'], 'playername'],
                    'team_index': d['team'],
                    'team_name': self.leagueteams.loc[d['team'], 'name'],
                }
                for (i, d) in enumerate(self.drafthistory, 1)
            ]
        )

    def save_history(self, fname):
        with open(fname, 'wb') as f:
            c = csv.DictWriter(f, fieldnames=self.drafthistory[0].keys())
            c.writeheader()
            c.writerows(self.drafthistory)

    def load_history(self, fname):
        with open(fname, 'rb') as f:
            return list(csv.DictReader(f))

    def draft_by_ids(self, pid, tid):
        player = self.data.loc[pid]
        if tid is None:
            team = None
        else:
            team = self.leagueteams.loc[tid]
        self.been_drafted(player, team, updateplots=False)

    def replay_draft(self, fname):
        hist = self.load_history(fname)

        for d in hist:
            pid = int(d['player'])
            tid = int(d['team'])
            self.draft_by_ids(pid, tid)
