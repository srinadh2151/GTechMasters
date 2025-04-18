""""""
"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Tucker Balch (replace with your name)
GT User ID: snidadana3 (replace with your User ID)
GT ID: 903966341 (replace with your GT ID)
"""

import datetime as dt
import numpy as np
import pandas as pd
import util as ut
from indicators import bollinger_bands, momentum, simple_moving_average, relative_strength_index
from marketsimcode import compute_portvals
from QLearner import QLearner

class StockData:
    """Holds stock information"""
    def __init__(self, symbol, start_date, end_date):
        self._symbol = symbol
        self._dates = pd.date_range(start_date, end_date)
        self._price = None
        self._high = None
        self._low = None
        self._volume = None
        self._fetch_data()

    @property
    def price(self):
        return self._price

    @property
    def high(self):
        return self._high

    @property
    def low(self):
        return self._low

    @property
    def volume(self):
        return self._volume

    @property
    def trading_dates(self):
        return self._price.index

    def _fetch_data(self):
        self._price = self._get_data('Adj Close')
        self._high = self._get_data('High')
        self._low = self._get_data('Low')
        self._volume = self._get_data('Volume')

    def _get_data(self, attribute):
        data = ut.get_data([self._symbol], self._dates, colname=attribute)
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
        return data[[self._symbol]]

class IndicatorDiscretizer:
    """Discretizes indicators"""
    @property
    def momentum_max_bucket(self):
        return 4

    @property
    def simple_moving_average_max_bucket(self):
        return 4

    @property
    def bollinger_bands_max_bucket(self):
        return 4

    @property
    def relative_strength_index_max_bucket(self):
        return 4

    def momentum(self, mtm):
        discretized = mtm.copy()
        discretized.values[mtm < -0.5] = 0
        discretized.values[(mtm >= -0.5) & (mtm <= 0.0)] = 1
        discretized.values[(mtm > 0.0) & (mtm <= 0.5)] = 2
        discretized.values[mtm > 0.5] = 3
        discretized.values[mtm.isnull()] = 4
        return discretized.astype('int32')

    def simple_moving_average(self, sma):
        discretized = sma.copy()
        discretized.values[sma < -0.5] = 0
        discretized.values[(sma >= -0.5) & (sma <= 0.0)] = 1
        discretized.values[(sma > 0.0) & (sma <= 0.5)] = 2
        discretized.values[sma > 0.5] = 3
        discretized.values[sma.isnull()] = 4
        return discretized.astype('int32')

    def bollinger_bands(self, bbands):
        discretized = bbands.copy()
        discretized.values[bbands < -1.0] = 0
        discretized.values[(bbands >= -1.0) & (bbands <= 0.0)] = 1
        discretized.values[(bbands > 0.0) & (bbands <= 1.0)] = 2
        discretized.values[bbands > 1.0] = 3
        discretized.values[bbands.isnull()] = 4
        return discretized.astype('int32')

    def relative_strength_index(self, rsi):
        discretized = rsi.copy()
        discretized.values[rsi < 30] = 0
        discretized.values[(rsi >= 30) & (rsi <= 70)] = 1
        discretized.values[rsi > 70] = 2
        discretized.values[rsi.isnull()] = 3
        return discretized.astype('int32')

class TradingStateFactory:
    """Factory that creates trading states from underlying technical indicators"""
    def __init__(self, stock_data, indicator_discretizer, lookback=10):
        self._stock_data = stock_data
        self._indicator_discretizer = indicator_discretizer
        self._lookback = lookback
        self._num_states = None
        self._indicators = None
        self._compute_number_of_states()
        self._compute_indicators()

    @property
    def num_states(self):
        return self._num_states

    def create(self, day):
        return self._indicators.loc[day]

    def _compute_number_of_states(self):
        all_buckets = [
            self._indicator_discretizer.momentum_max_bucket,
            self._indicator_discretizer.simple_moving_average_max_bucket,
            self._indicator_discretizer.bollinger_bands_max_bucket,
            self._indicator_discretizer.relative_strength_index_max_bucket
        ]
        largest_number = int(''.join(map(str, all_buckets)))
        self._num_states = largest_number + 1

    def _compute_indicators(self):
        price = self._stock_data.price
        high = self._stock_data.high
        low = self._stock_data.low
        volume = self._stock_data.volume

        mtm = momentum(price, self._lookback)
        sma, sma_ratio = simple_moving_average(price, self._lookback)
        bbands = bollinger_bands(price, sma, self._lookback)
        rsi = relative_strength_index(price, self._lookback)

        self._indicators = self._discretize((mtm, sma_ratio, bbands, rsi))

    def _discretize(self, indicators):
        mtm, sma, bbands, rsi = indicators

        discretized_mtm = self._indicator_discretizer.momentum(mtm)
        discretized_sma = self._indicator_discretizer.simple_moving_average(sma)
        discretized_bbands = self._indicator_discretizer.bollinger_bands(bbands)
        discretized_rsi = self._indicator_discretizer.relative_strength_index(rsi)

        discretized_indicators = pd.concat(
            [discretized_mtm, discretized_sma, discretized_bbands, discretized_rsi],
            axis=1
        )

        discretized_indicators = discretized_indicators.apply(
            lambda row: int(''.join(map(str, row))), axis=1
        )

        return discretized_indicators

class TradingEnvironment:
    """Encapsulates trading as a Reinforcement Learning environment"""
    def __init__(self):
        self._qlearner = None
        self._trading_state_factory = None
        self._stock_data = None
        self._trading_options = None
        self._action_mapping = {0: 'LONG', 1: 'CASH', 2: 'SHORT'}

    @property
    def qlearner(self):
        return self._qlearner

    @qlearner.setter
    def qlearner(self, qlearner):
        self._qlearner = qlearner

    @property
    def trading_state_factory(self):
        return self._trading_state_factory

    @trading_state_factory.setter
    def trading_state_factory(self, trading_state_factory):
        self._trading_state_factory = trading_state_factory

    @property
    def stock_data(self):
        return self._stock_data

    @stock_data.setter
    def stock_data(self, stock_data):
        self._stock_data = stock_data

    @property
    def trading_options(self):
        return self._trading_options

    @trading_options.setter
    def trading_options(self, trading_options):
        self._trading_options = trading_options

    def run_learning_episode(self):
        holding = None
        trading_dates = self._trading_options['trading_dates']
        orders = pd.DataFrame(index=trading_dates, columns=['Shares'])

        state = self._trading_state_factory.create(trading_dates[0])
        self._qlearner.querysetstate(state)

        for index, date in enumerate(trading_dates):
            yesterday = trading_dates[index - 1] if index > 0 else date
            reward = self._reward(date, yesterday, holding)
            action = self._qlearner.query(state, reward)
            order, holding = self._execute_action(action, holding)
            orders.loc[date] = order

            if index == len(trading_dates) - 1:
                return orders

            state = self._trading_state_factory.create(trading_dates[index + 1])

    def run_interaction_episode(self):
        holding = None
        trading_dates = self._trading_options['trading_dates']
        orders = pd.DataFrame(index=trading_dates, columns=['Shares'])

        for index, date in enumerate(trading_dates):
            state = self._trading_state_factory.create(date)
            action = self._qlearner.querysetstate(state)
            order, holding = self._execute_action(action, holding)
            orders.loc[date] = order

        return orders

    def _reward(self, today, yesterday, holding):
        if holding == 'CASH' or holding is None:
            return 0.

        price_today = self._apply_impact(self._stock_data.price.loc[today], holding)
        price_yesterday = self._stock_data.price.loc[yesterday]
        daily_return = (price_today / price_yesterday) - 1.
        multiplier = 1. if holding == 'LONG' else -1.

        return daily_return * multiplier

    def _apply_impact(self, price, holding):
        impact = self._trading_options['impact']

        if holding == 'LONG':
            return price * (1. - impact)
        if holding == 'SHORT':
            return price * (1. + impact)

        return price

    def _execute_action(self, action, holding):
        action_label = self._action_mapping[action]

        if action_label == 'LONG':
            return self._execute_long(holding)
        elif action_label == 'SHORT':
            return self._execute_short(holding)
        elif action_label == 'CASH':
            return self._execute_cash(holding)
        else:
            raise ValueError("Unrecognized action: {}".format(action_label))

    def _execute_long(self, holding):
        if holding == 'LONG':
            return 0, 'LONG'

        to_buy = 1000
        if holding == 'SHORT':
            to_buy = 2000

        return to_buy, 'LONG'

    def _execute_short(self, holding):
        if holding == 'SHORT':
            return 0, 'SHORT'

        to_sell = 1000
        if holding == 'LONG':
            to_sell = 2000

        return to_sell * -1, 'SHORT'

    def _execute_cash(self, holding):
        if holding == 'LONG':
            return -1000, 'CASH'
        if holding == 'SHORT':
            return 1000, 'CASH'

        return 0, 'CASH'

class StrategyLearner:
    def __init__(self, verbose=False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self._learner = None
        self._indicator_discretizer = IndicatorDiscretizer()
        self._trading_environment = TradingEnvironment()
        self._metadata = {}

    @property
    def metadata(self):
        return self._metadata

    def add_evidence(self, symbol="IBM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=10000):
        stock_data = StockData(symbol, sd, ed)
        trading_state_factory = TradingStateFactory(stock_data, self._indicator_discretizer)

        self._learner = QLearner(
            num_states=trading_state_factory.num_states,
            num_actions=3,
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=0
        )

        self._trading_environment.qlearner = self._learner
        self._trading_environment.trading_state_factory = trading_state_factory
        self._trading_environment.stock_data = stock_data
        self._trading_environment.trading_options = {
            'trading_dates': stock_data.trading_dates,
            'impact': self.impact
        }

        latest_cumulative_return = -999
        current_cumulative_return = 0
        episodes = 0

        while np.abs(latest_cumulative_return - current_cumulative_return) > 0.001:
            latest_cumulative_return = current_cumulative_return

            trades = self._trading_environment.run_learning_episode()
            orders = self._convert_trades_to_marketisim_orders(symbol, trades)

            portfolio_values = compute_portvals(
                orders,
                start_val=sv,
                commission=0.,
                impact=self.impact,
                prices=stock_data.price.copy(),
            )

            current_cumulative_return = self._compute_cumulative_return(portfolio_values)
            episodes += 1

        self._metadata['training_episodes'] = episodes

    def testPolicy(self, symbol = "IBM", sd=dt.datetime(2009,1,1), ed=dt.datetime(2010,1,1), sv = 10000):

        stock_data = StockData(symbol, sd, ed)
        trading_state_factory = TradingStateFactory(stock_data, self._indicator_discretizer)

        self._trading_environment.trading_state_factory = trading_state_factory
        self._trading_environment.stock_data = stock_data
        self._trading_environment.trading_options = {
            'trading_dates': stock_data.trading_dates,
            'impact': self.impact
        }

        trades = self._trading_environment.run_interaction_episode()

        # Keep track of the total number of entries generated
        self._metadata['entries'] = self._count_total_number_of_entries(trades)

        return trades

    def _convert_trades_to_marketisim_orders(self, symbol, trades):
        # Convert the trades into the format expected by my marketsimcode.py
        orders = pd.DataFrame(index=trades.index, columns=['Order', 'Date', 'Symbol', 'Shares'])

        for index, trade in trades.iterrows():
            shares = trade['Shares']

            if shares == 0:
                orders.loc[index] = ['HOLD', index, symbol, shares]
            elif shares > 0:
                orders.loc[index] = ['BUY', index, symbol, shares]
            else:
                orders.loc[index] = ['SELL', index, symbol, shares * -1]

        return orders

    def _compute_cumulative_return(self, portfolio_values):
        return (portfolio_values[-1] / portfolio_values[0]) - 1

    def _count_total_number_of_entries(self, trades):
        # Entries are any trades were the strategy suggests
        # either going long (positive) or shorting (negative)
        return trades.values[trades != 0].shape[0]

def author():
    return 'snidadana3'

if __name__=="__main__":
    print ("One does not simply think up a strategy")