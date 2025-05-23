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
import logging
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os

# print('Current Working Dir:', os.getcwd())  # Change the current working directory to the project root
# os.chdir('./CS7646 ML4T/ML4T_2025Spring/strategy_evaluation')  # Change to the correct directory
print('Current Working Dir:', os.getcwd())  # Change the current working directory to the project root

from marketsimcode import compute_portvals
from StrategyLearner import StrategyLearner

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    """
    Conduct an experiment with StrategyLearner to show how changing the value of impact
    affects in-sample trading behavior. Use two metrics: number of trades and cumulative return.
    """

    # Hypothesis:
    # The market impact parameter represents the cost of trading, which affects the strategy's
    # aggressiveness. As the impact increases, the StrategyLearner is expected to become more
    # conservative, resulting in fewer trades. This is because the cost of trading reduces the
    # potential profit from frequent trades. Consequently, the cumulative return is also expected
    # to decrease with higher impact values, as the strategy avoids trades that are not profitable
    # after accounting for the impact cost.

    # Set the seed for reproducibility
    random.seed(7181090000)

    # Experiment parameters
    symbol = 'JPM'
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    starting_value = 100000
    commission = 0.0
    impact_values = [0.0, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]  # Impact values to test

    # Lists to store results
    all_trades_count = []
    all_cumulative_returns = []
    all_entries = []
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

        # Count the number of trades
        num_trades = trades[trades != 0].count()
        all_trades_count.append(num_trades)

        # Compute portfolio values
        orders = _convert_trades_to_marketisim_orders(symbol, trades)
        port_vals = compute_portvals(
            orders,
            start_val=starting_value,
            commission=commission,
            impact=impact
        )

        # Calculate cumulative return
        cumulative_return = _compute_cumulative_return(port_vals)
        all_cumulative_returns.append(cumulative_return)

        # Collect additional metrics
        all_entries.append(strategy_learner.metadata['entries'])
        all_episodes.append(strategy_learner.metadata['training_episodes'])

    # Plot and save results
    _plot_and_save_results(impact_values, all_trades_count, all_cumulative_returns)
    _plot_and_save_number_of_entries_per_impact_value(impact_values, all_entries)
    _plot_and_save_number_of_episodes_per_impact_value(impact_values, all_episodes)
    _plot_and_save_cumulative_return_per_impact_value(impact_values, all_cumulative_returns)

def author():
    """
    Returns the author's GT username.
    """
    return 'snidadana3'

def _convert_trades_to_marketisim_orders(symbol, trades):
    """
    Convert trades DataFrame to the format expected by marketsimcode.py.
    """
    # Ensure the DataFrame has the correct columns
    orders = pd.DataFrame(index=trades.index, columns=['Order', 'Date', 'Symbol', 'Shares'])

    for index, trade in trades.iterrows():
        shares = trade['Shares']
        if shares == 0:
            orders.loc[index] = ['HOLD', index, symbol, shares]
        elif shares > 0:
            orders.loc[index] = ['BUY', index, symbol, shares]
        else:
            orders.loc[index] = ['SELL', index, symbol, -shares]  # Corrected to match the expected shape

    return orders

def _compute_cumulative_return(portfolio_values):
    """
    Compute the cumulative return of the portfolio.
    """
    return (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1

def _plot_and_save_results(impact_values, trades_count, cumulative_returns):
    """
    Plot and save the number of trades and cumulative returns for different impact values.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot number of trades
    ax1.set_xlabel('Impact Value')
    ax1.set_ylabel('Number of Trades', color='tab:blue')
    ax1.plot(impact_values, trades_count, 'o-', color='tab:blue', label='Number of Trades')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot cumulative returns
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cumulative Return', color='tab:red')
    ax2.plot(impact_values, cumulative_returns, 's-', color='tab:red', label='Cumulative Return')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Title and layout
    plt.title('Impact of Market Impact on Trading Behavior')
    fig.tight_layout()

    # Save the plot
    plt.savefig('./images/impact_on_trading_behavior.png')
    plt.show()

    # Explanation of results
    log.info("The plot shows how the number of trades and cumulative returns change with different impact values.")
    log.info("As the impact value increases, the number of trades generally decreases, indicating a more conservative strategy.")
    log.info("Similarly, the cumulative return tends to decrease with higher impact values, reflecting the cost of trading.")

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

def _generate_bar_plot(data, title, xlabel, ylabel, bar_label, groups, filename):
    log.info("Generating bar plot with title '%s'", title)

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

    plt.xticks(index + bar_width / 2, groups, rotation=45, ha='right')

    plt.legend()
    plt.tight_layout()

    # save_folder = './images'
    # print('save_folder:', os.getcwd())
    save_folder = './'
    plt.savefig(f"{save_folder}/{filename}.png", bbox_inches='tight')

    log.info("Saved bar plot to file: %s", filename)

if __name__ == '__main__':
    main()