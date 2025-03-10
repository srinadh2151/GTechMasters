import pandas as pd
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
# from util import get_data
import util as ut
from indicators import bollinger_bands, simple_moving_average, relative_strength_index, momentum, commodity_channel_index
from TheoreticallyOptimalStrategy import testPolicy
import marketsimcode as mm

def author():
    return 'snidadana3'

def get_indicators(prices):
    """Calculate and return all indicators as a DataFrame."""
    indicators = {} #pd.DataFrame(index=prices.index)
    indicators['SMA'] = simple_moving_average(prices)
    indicators['bb_value'] = bollinger_bands(prices)
    indicators['RSI'] = relative_strength_index(prices)
    indicators['momentum'] = momentum(prices)
    indicators['CCI'] = commodity_channel_index(prices)
    return indicators

def save_plot(fig, title):
    """Save the plot to the images directory."""
    if not os.path.exists('images'):
        os.makedirs('images')
    fig.savefig(f'./images/{title}.png')
    print(f'Plot saved to images/{title}.png')

def run_test_policy():
    """Run the test policy and plot results."""
    symbol = 'JPM'
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    sv = 100000

    # Run the theoretically optimal strategy
    orders = testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    # print('Orders:', orders)

    # Compute portfolio values
    portvals = mm.compute_portvals(orders, start_val=sv) # symbols=[symbol],

    # Normalize portfolio values
    portvals = portvals / portvals.iloc[0]

    # Plot portfolio values
    plt.figure(figsize=(10, 6))
    plt.plot(portvals, label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.show()
    save_plot(plt, 'Portfolio Value Over Time')
    print('Portfolio Value Plot Saved')

def main():
    # Define date range
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    symbol = 'JPM'

    # Get stock data
    dates = pd.date_range(sd, ed)
    prices_all = ut.get_data([symbol], dates)    
    prices = prices_all[symbol]
    print('Prices Started:')

    # Calculate indicators
    indicators = get_indicators(prices)

    # Plot indicators

    # Plot indicators
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    # plt.figure(figsize=(14, 10))

    # Bollinger Bands
    ax = axs[0, 0]
    bb_df = indicators['bb_value']
    ax.plot(prices.index, prices, label='Price')
    ax.plot(prices.index, bb_df['rolling mean'], label='Rolling Mean', linestyle='-')
    ax.plot(prices.index, bb_df['upper band'], label='Upper Band', linestyle='-')
    ax.plot(prices.index, bb_df['lower band'], label='Lower Band', linestyle='-')
    ax.plot(prices.index, bb_df['bb value'], label='BB Value', linestyle=':')
    ax.set_title('Bollinger Bands')
    ax.legend()

    # Simple Moving Average
    ax = axs[0, 1]
    sma_df = simple_moving_average(prices)
    ax.plot(prices.index, prices, label='Price')
    ax.plot(prices.index, sma_df['simple moving average'], label='SMA', linestyle='-')
    ax.set_title('Simple Moving Average')
    ax.legend()

    # Relative Strength Index
    ax = axs[1, 0]
    ax.plot(prices.index, prices, label='Price', linestyle='-', alpha=0.5)
    ax.plot(prices.index, relative_strength_index(prices), label='RSI')
    ax.set_title('Relative Strength Index')
    ax.legend()

    # Momentum
    ax1 = axs[1, 1]
    ax2 = ax1.twinx()
    ax1.plot(prices.index, prices, label='Price', linestyle='-', alpha=0.5, color='b')
    ax2.plot(prices.index, momentum(prices), label='Momentum', color='r')
    ax1.set_title('Momentum')
    ax1.set_ylabel('Price', color='b')
    ax2.set_ylabel('Momentum', color='r')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Commodity Channel Index
    ax = axs[2, 0]
    cci_df = commodity_channel_index(prices)
    ax.plot(prices.index, cci_df['Commodity Channel Index'], label='CCI')
    ax.plot(prices.index, cci_df['Normalized Price'], label='Normalized Price', linestyle='-')
    ax.set_title('Commodity Channel Index')
    ax.legend()

    plt.tight_layout()
    save_plot(fig, f'Indicators_{symbol}')

    # Run the test policy
    run_test_policy()

if __name__ == "__main__":
    main()