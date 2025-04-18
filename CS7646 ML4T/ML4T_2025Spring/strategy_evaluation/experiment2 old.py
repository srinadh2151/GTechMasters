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
import json
import logging
import numpy as np
import pandas as pd
import random
import sys

from marketsimcode import compute_portvals
from StrategyLearner import StrategyLearner

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    """
    Provide an hypothesis regarding how changing the value of impact
    should affect in sample trading behavior and results (provide at
    least two metrics). Conduct an experiment with JPM on the
    in sample period to test that hypothesis. Provide charts, graphs
    or tables that illustrate the results of your experiment.
    """

    # Hypothesis:
    # The market impact parameter represents the cost of trading, which affects the strategy's
    # aggressiveness. As the impact increases, the StrategyLearner is expected to become more
    # conservative, resulting in fewer trades. This is because the cost of trading reduces the
    # potential profit from frequent trades. Consequently, the cumulative return is also expected
    # to decrease with higher impact values, as the strategy avoids trades that are not profitable
    # after accounting for the impact cost.

    # Set the seed for reproducibility
    random.seed(5210780000)

    # Experiment parameters
    symbol = 'JPM'
    # In-sample: January 1, 2008 to December 31 2009
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    starting_value = 100000
    commission = 0.0
    # Values to use to evaluate the effect of the impact
    impact_values = [0.0, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

    all_entries = []
    all_returns = []
    all_episodes = []

    for impact in impact_values:
        log.info("Evaluating the effect of impact=%s", impact)
        strategy_learner = StrategyLearner(verbose=False, impact=impact)

        # Train the StrategyLearner
        strategy_learner.add_evidence(
            symbol=symbol,
            sd=start_date,
            ed=end_date,
            sv=starting_value
        )

        # Generate trades using the trained StrategyLearner
        trades = strategy_learner.testPolicy(
            symbol=symbol,
            sd=start_date,
            ed=end_date,
            sv=starting_value
        )

        log.info("Converting StrategyLearner trades to marketsim orders")
        orders = _convert_trades_to_marketisim_orders(symbol, trades)

        log.info("Computing portfolio values for %d orders", orders.shape[0])
        port_vals = compute_portvals(
            orders,
            start_val=starting_value,
            commission=commission,
            impact=impact
        )

        cumulative_return = _compute_cumulative_return(port_vals)

        all_entries.append(strategy_learner.metadata['entries'])
        all_returns.append(cumulative_return)
        all_episodes.append(strategy_learner.metadata['training_episodes'])

    _plot_and_save_number_of_entries_per_impact_value(impact_values, all_entries)
    _plot_and_save_number_of_episodes_per_impact_value(impact_values, all_episodes)
    _plot_and_save_cumulative_return_per_impact_value(impact_values, all_returns)

def author():
    """
    Returns the author's GT username.
    """
    return 'snidadana3'

def _plot_and_save_number_of_entries_per_impact_value(impact_values, entries):
    _generate_bar_plot(
        entries,
        'Number of entries per impact value - In Sample',
        'Impact value',
        'Number of entries',
        'Entries',
        impact_values,
        'Experiment2-NumberOfEntries'
    )

    _save_as_json(impact_values, entries, 'entries_per_impact')

def _plot_and_save_number_of_episodes_per_impact_value(impact_values, episodes):
    _generate_bar_plot(
        episodes,
        'Number of training episodes per impact value - In Sample',
        'Impact value',
        'Number of training episodes',
        'Episodes',
        impact_values,
        'Experiment2-NumberOfEpisodes'
    )

    _save_as_json(impact_values, episodes, 'episodes_per_impact')

def _plot_and_save_cumulative_return_per_impact_value(impact_values, returns):
    _generate_bar_plot(
        returns,
        'Cumulative return per impact value - In Sample',
        'Impact value',
        'Cumulative return (%)',
        'Cumulative return',
        impact_values,
        'Experiment2-CumulativeReturn'
    )

    _save_as_json(impact_values, returns, 'cumulative_return_per_impact')

def _convert_trades_to_marketisim_orders(symbol, trades):
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

def _compute_cumulative_return(portfolio_values):
    return (portfolio_values[-1] / portfolio_values[0]) - 1

def _generate_bar_plot(data, title, xlabel, ylabel, bar_label, groups, filename):
    # See: https://matplotlib.org/examples/pylab_examples/barchart_demo.html
    log.info("Generating bar plot with title '%s'", title)

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.close()

    _, ax = plt.subplots()
    ax.grid(color='black', linestyle='dotted')

    index = np.arange(len(groups))
    bar_width = 0.35
    opacity = 0.4

    bar = plt.bar(index, data, bar_width, alpha=opacity, color='b', label=bar_label)

    plt.xlabel(xlabel, fontsize='15')
    plt.ylabel(ylabel, fontsize='15')
    plt.title(title)

    # Rotate tick labels and align them
    # See: https://stackoverflow.com/questions/14852821/aligning-rotated-xticklabels-with-their-respective-xticks
    # See: https://matplotlib.org/examples/ticks_and_spines/ticklabels_demo_rotation.html
    plt.xticks(index + bar_width / 2, groups, rotation=45, ha='right')

    plt.legend()
    plt.tight_layout()

    # save_folder = './CS7646 ML4T/ML4T_2025Spring/strategy_evaluation/images'
    save_folder = './'
    
    plt.savefig(f"{save_folder}/{filename}.png", bbox_inches='tight')

    log.info("Saved bar plot to file: %s", filename)

def _save_as_json(impact_values, metrics, filename):
    log.info("Creating JSON data")

    data = {}
    for index, impact in enumerate(impact_values):
        data[impact] = metrics[index]

    # save_folder = './CS7646 ML4T/ML4T_2025Spring/strategy_evaluation/images'
    save_folder = './'
    
    filepath = f"{save_folder}/{filename}.json"
    
    with open(filepath, 'w') as outfile:
        json.dump(data, outfile, indent=2, separators=(',', ': '))

    log.info("JSON data saved to file: %s", filename)

if __name__ == '__main__':
    # Configure our logger
    log = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    )
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False

    main()