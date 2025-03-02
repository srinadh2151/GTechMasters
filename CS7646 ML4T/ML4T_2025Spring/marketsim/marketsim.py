""""""
"""MC2-P1: Market simulator.

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
import os

import numpy as np

import pandas as pd
# os.chdir('../../')
# os.getcwd()
from util import get_data, plot_data


def compute_portvals(
    orders_file="./orders/orders.csv",
    # orders_file="./marketsim/orders/orders.csv",
    start_val=1000000,
    commission=9.95,
    impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    # start_date = dt.datetime(2008, 1, 1)
    # end_date = dt.datetime(2008, 6, 1)
    # portvals = get_data(["IBM"], pd.date_range(start_date, end_date))
    # portvals = portvals[["IBM"]]  # remove SPY
    # rv = pd.DataFrame(index=portvals.index, data=portvals.values)

    # Read orders file
    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    orders.sort_index(inplace=True)

    # Get the date range for the simulation
    start_date = orders.index.min()
    end_date = orders.index.max()

    # Get the list of symbols
    symbols = orders['Symbol'].unique().tolist()

    # Get stock data
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices['Cash'] = 1.0  # Add a cash column for cash transactions

    # Initialize trades and holdings dataframes
    trades = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    holdings = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    holdings.iloc[0, -1] = start_val  # Set initial cash


    # Process each order
    for date, order in orders.iterrows():
        symbol = order['Symbol']
        shares = order['Shares']
        order_type = order['Order']

        # Calculate the price with market impact
        price = prices.loc[date, symbol]
        if order_type == 'BUY':
            price *= (1 + impact)
            trades.loc[date, symbol] += shares
            trades.loc[date, 'Cash'] -= (price * shares + commission)
        elif order_type == 'SELL':
            price *= (1 - impact)
            trades.loc[date, symbol] -= shares
            trades.loc[date, 'Cash'] += (price * shares - commission)

    # Calculate holdings
    holdings = trades.cumsum()
    # holdings.iloc[0, -1] += start_val  # Ensure initial cash is added to holdings

    # Calculate portfolio values
    portvals = (holdings * prices).sum(axis=1).round(2)
    
    portvals_df = pd.DataFrame(portvals, columns=['Portfolio Value'])
    print('****portvals_df ---\n', portvals_df)

    return portvals_df

    # return rv
    # return portvals

def author():
    return 'snidadana3'  # replace with your Georgia Tech username

def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters
    # print('current working directory:', os.getcwd(), '\n', os.listdir())
    of = "./orders/orders2.csv"
    of = "./CS7646 ML4T/ML4T_2025Spring/marketsim/orders/orders-02.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2008, 6, 1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
        0.2,
        0.01,
        0.02,
        1.5,
    ]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
        0.2,
        0.01,
        0.02,
        1.5,
    ]

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()
