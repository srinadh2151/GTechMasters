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
os.environ["ORDERS_DATA_DIR"] = './CS7646 ML4T/ML4T_2025Spring/marketsim/orders'

def compute_portvals(
    trades_df,
    # symbols,
    start_val=1000000,
    commission=9.95,
    impact=0.005,
):
    """
    Computes the portfolio values.

    :param trades_df: DataFrame containing trade orders with columns ['Date', 'Symbol', 'Order', 'Shares']
    :type trades_df: pandas.DataFrame
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # Ensure the DataFrame is sorted by date
    trades_df.sort_index(inplace=True)

    # Get the date range for the simulation
    start_date = trades_df.index.min()
    end_date = trades_df.index.max()

    # Get the list of symbols
    symbols = trades_df['Symbol'].unique().tolist()
    # We changed the logic to get all symbols from input
    # symbols = ['JPM']

    # Get stock data
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices['Cash'] = 1.0  # Add a cash column for cash transactions

    # Initialize trades and holdings dataframes
    trades = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    holdings = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    holdings.iloc[0, -1] = start_val  # Set initial cash

    # Process each order
    for date, order in trades_df.iterrows():
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

    # Calculate portfolio values
    portvals = (holdings * prices).sum(axis=1)
    
    portvals_df = pd.DataFrame(portvals, columns=['Portfolio Value'])
    portvals_df['Portfolio Value'] = portvals_df['Portfolio Value'] + start_val

    return portvals_df

def compute_portvals_new(orders_file, start_val = 100000, commission=9.95, impact=0.005):
    """
    Computes the portfolio values.

    :param trades_df: DataFrame containing trade orders with columns ['Date', 'Symbol', 'Order', 'Shares']
    :type trades_df: pandas.DataFrame
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    
    dates = orders_file.index
    symbol = orders_file.columns[0]
    
    # prices data is the Adj close price per trading day
    prices_data = get_data([symbol], pd.date_range(dates[0],dates[-1]))
     # SPY is kept to distinguish trading days, removed if not in the portfolio, get_data adds it automatically
    if symbol != 'SPY':
        prices_data = prices_data.drop('SPY', axis=1)
        
    # df_prices is price data with the cash feature
    df_prices = pd.DataFrame(prices_data)
    df_prices['cash'] = 1
    
    # df_trades represents number of shares held and cash avalable only on order dates
    df_trades = orders_file.copy()
    
    # df_holdings represents df_trades, but on days inbetween traded days
    df_holdings = df_trades.copy()
        
    for i in orders_file.index:
        if orders_file.loc[i,symbol] != 0: # prevents transaction costs on non-trading days
            total_cost = orders_file.loc[i, symbol] * df_prices.loc[i, symbol] # to clean up the code
            df_trades.loc[i, 'cash'] = -total_cost - abs(commission + total_cost * impact) 
    df_trades.fillna(0, inplace=True)
    
    df_holdings.loc[dates[0],'cash'] = start_val + df_trades.loc[dates[0],'cash']
    df_holdings.iloc[0, :-1] = df_trades.iloc[0, :-1]
    
    for i in range(1, df_holdings.shape[0]):
        df_holdings.iloc[i, :] = df_trades.iloc[i, :] + df_holdings.iloc[i-1, :]
        
    # df_value is the dollar value of the shares at each date
    df_value = df_holdings.multiply(df_prices)
    
    df_portval = df_value.sum(axis=1)
    
    return df_portval
    
def author():
    return 'snidadana3'

def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters
    # print('current working directory:', os.getcwd(), '\n', os.listdir())
    # of = "./orders/orders2.csv"
    of = "./CS7646 ML4T/ML4T_2025Spring/marketsim/orders/orders-02.csv"
    sv = 1000000
    
    # Load the Data    
    orders_df = pd.read_csv(of, index_col='Date', parse_dates=True, na_values=['nan'])

    # Process orders
    portvals = compute_portvals(trades_df= orders_df, start_val= sv)
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
