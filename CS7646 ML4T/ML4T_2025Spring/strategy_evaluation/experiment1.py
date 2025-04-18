""""""
"""
Template for implementing experiment1  (c) 2016 Tucker Balch

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
import pandas as pd
import sys
import os

from ManualStrategy import ManualStrategy
from marketsimcode import compute_portvals
from StrategyLearner import StrategyLearner

# save_folder = './CS7646 ML4T/ML4T_2025Spring/strategy_evaluation/'
# print('Current Directory:', os.getcwd())
# os.chdir('./CS7646 ML4T/ML4T_2025Spring/strategy_evaluation')

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    """
    Compare the manual strategy with the learning strategy in-sample.
    Plot the performance of both strategies along with the benchmark.
    Trade only the symbol JPM for this evaluation.
    """

    # In-sample experiment parameters
    in_sample_params = {
        'symbol': 'JPM',
        'start_date': dt.datetime(2008, 1, 1),
        'end_date': dt.datetime(2009, 12, 31),
        'starting_value': 100000,
        'commission': 0.0,  # Commission cost
        'impact': 0.0       # Market impact
    }

    # Out-of-sample experiment parameters
    out_sample_params = {
        'symbol': 'JPM',
        'start_date': dt.datetime(2010, 1, 1),
        'end_date': dt.datetime(2011, 12, 31),
        'starting_value': 100000,
        'commission': 0.0,  # Commission cost
        'impact': 0.0       # Market impact
    }

    # Evaluate in-sample strategies and benchmark
    in_manual_strategy_values = _evaluate_manual_strategy(in_sample_params)
    in_strategy_learner_values = _evaluate_strategy_learner(in_sample_params)
    in_benchmark_values = _evaluate_benchmark(in_sample_params, in_strategy_learner_values.index)

    # Normalize in-sample portfolio values
    in_manual_strategy_values = _normalize(in_manual_strategy_values)
    in_strategy_learner_values = _normalize(in_strategy_learner_values)
    in_benchmark_values = _normalize(in_benchmark_values)

    # Plot in-sample results
    _plot(
        [
            (in_strategy_learner_values.index, in_strategy_learner_values),
            (in_manual_strategy_values.index, in_manual_strategy_values),
            (in_benchmark_values.index, in_benchmark_values)
        ],
        'StrategyLearner vs ManualStrategy vs Benchmark - In Sample',
        'Date',
        'Normalized Value',
        'Experiment1_InSample',
        colors=['blue', 'red', 'black'],
        legend=['StrategyLearner', 'ManualStrategy', 'Benchmark']
    )
    
    # Evaluate out-of-sample strategies and benchmark
    out_manual_strategy_values = _evaluate_manual_strategy(out_sample_params)
    out_strategy_learner_values = _evaluate_strategy_learner(out_sample_params)
    out_benchmark_values = _evaluate_benchmark(out_sample_params, out_strategy_learner_values.index)

    # Normalize out-of-sample portfolio values
    out_manual_strategy_values = _normalize(out_manual_strategy_values)
    out_strategy_learner_values = _normalize(out_strategy_learner_values)
    out_benchmark_values = _normalize(out_benchmark_values)

    # Plot out-of-sample results
    _plot(
        [
            (out_strategy_learner_values.index, out_strategy_learner_values),
            (out_manual_strategy_values.index, out_manual_strategy_values),
            (out_benchmark_values.index, out_benchmark_values)
        ],
        'StrategyLearner vs ManualStrategy vs Benchmark - Out of Sample',
        'Date',
        'Normalized Value',
        'Experiment1_OutSample',
        colors=['blue', 'red', 'black'],
        legend=['StrategyLearner', 'ManualStrategy', 'Benchmark']
    )

def author():
    """
    Returns the author's GT username.
    """
    return 'snidadana3'

def _evaluate_manual_strategy(params):
    """
    Evaluate the manual strategy using the given parameters.
    """
    log.info("Evaluating ManualStrategy using params: %s", params)

    # Extract parameters
    symbol = params['symbol']
    start_date = params['start_date']
    end_date = params['end_date']
    starting_value = params['starting_value']
    commission = params['commission']
    impact = params['impact']

    # Initialize and query the manual strategy
    manual_strategy = ManualStrategy()
    log.info("Querying ManualStrategy to generate orders")
    orders = manual_strategy.testPolicy(
        symbol,
        start_date,
        end_date,
        starting_value
    )

    # Compute portfolio values
    log.info("Computing portfolio values for %d orders", orders.shape[0])
    port_vals = compute_portvals(
        orders,
        start_val=starting_value,
        commission=commission,
        impact=impact
    )

    # Calculate cumulative return
    cumulative_return = _get_portfolio_performance(port_vals)
    log.info("ManualStrategy stats: final value=%s, cumulative return=%s",
             port_vals.iloc[-1], cumulative_return)

    return port_vals

def _evaluate_strategy_learner(params):
    """
    Evaluate the strategy learner using the given parameters.
    """
    log.info("Evaluating StrategyLearner using params: %s", params)

    # Extract parameters
    symbol = params['symbol']
    start_date = params['start_date']
    end_date = params['end_date']
    starting_value = params['starting_value']
    commission = params['commission']
    impact = params['impact']

    # Initialize and train the strategy learner
    strategy_learner = StrategyLearner(verbose=False, impact=impact)
    log.info("Training StrategyLearner")
    strategy_learner.add_evidence(
        symbol,
        start_date,
        end_date,
        starting_value
    )

    # Query the strategy learner to generate trades
    log.info("Querying StrategyLearner to generate trades")
    trades = strategy_learner.testPolicy(
        symbol,
        start_date,
        end_date,
        starting_value
    )

    # Convert trades to market simulation orders
    log.info("Transforming StrategyLearner trades into marketsim orders")
    orders = _convert_trades_to_marketisim_orders(symbol, trades)

    # Compute portfolio values
    log.info("Computing portfolio values for %d orders", orders.shape[0])
    port_vals = compute_portvals(
        orders,
        start_val=starting_value,
        commission=commission,
        impact=impact
    )

    # Calculate cumulative return
    cumulative_return = _get_portfolio_performance(port_vals)
    log.info("StrategyLearner stats: final value=%s, cumulative return=%s",
             port_vals.iloc[-1], cumulative_return)

    return port_vals

def _evaluate_benchmark(params, trading_dates):
    """
    Evaluate the benchmark strategy using the given parameters.
    """
    log.info("Evaluating benchmark using params: %s", params)

    # Extract parameters
    symbol = params['symbol']
    starting_value = params['starting_value']
    commission = params['commission']
    impact = params['impact']

    # Generate buy-and-hold orders for the benchmark
    log.info("Generating orders for benchmark")
    orders = pd.DataFrame(index=trading_dates, columns=['Order', 'Date', 'Symbol', 'Shares'])
    orders.iloc[0] = ['BUY', trading_dates[0], symbol, 1000]
    orders.iloc[1:] = [['HOLD', date, symbol, 0] for date in trading_dates[1:]]

    # Compute portfolio values
    log.info("Computing portfolio values for %d orders", orders.shape[0])
    port_vals = compute_portvals(
        orders,
        start_val=starting_value,
        commission=commission,
        impact=impact
    )

    # Calculate cumulative return
    cumulative_return = _get_portfolio_performance(port_vals)
    log.info("Benchmark stats: final value=%s, cumulative return=%s",
             port_vals.iloc[-1], cumulative_return)

    return port_vals

def _convert_trades_to_marketisim_orders(symbol, trades):
    """
    Convert trades into the format expected by the market simulation code.
    """
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

def _get_portfolio_performance(portfolio_values):
    """
    Calculate the cumulative return of the portfolio.
    """
    return (portfolio_values[-1] / portfolio_values[0]) - 1

def _normalize(values):
    """
    Normalize the portfolio values to start at 1.0.
    """
    return values / values.iloc[0]

def _plot(data, title, xlabel, ylabel, filename, colors=['b', 'r', 'g'], legend=None):
    """
    Generate and save a plot of the given data.
    """
    log.info("Generating plot with title '%s'", title)

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.close()

    fig, ax = plt.subplots()
    fig.autofmt_xdate()  # Fix display issues with dates

    ax.grid(color='black', linestyle='dotted')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # Plot each dataset
    [plt.plot(
        xdata,
        ydata,
        linewidth=2.5,
        color=colors[index],
        alpha=0.4,
        label=legend[index])
     for index, (xdata, ydata) in enumerate(data)]

    plt.title(title)
    plt.xlabel(xlabel, fontsize='15')
    plt.ylabel(ylabel, fontsize='15')

    if legend is not None:
        plt.legend(fontsize='small')
        
    plt.savefig(f"./images/{filename}.png", bbox_inches='tight')

    log.info("Saved plot to file: %s", filename)

if __name__ == '__main__':
    # Configure logger for console output
    log = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    )
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False

    main()
