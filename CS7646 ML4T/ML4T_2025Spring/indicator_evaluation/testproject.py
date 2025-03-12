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
    
    print('orders data types - ', orders.dtypes)

    # Compute portfolio values
    portvals = mm.compute_portvals(orders, start_val=sv) # symbols=[symbol],

    # Normalize portfolio values
    portvals = portvals / portvals.iloc[0]
    
    print('portvals\n', portvals)
    # portvals.to_csv('portvals.csv', index=False)
    
    # Create benchmark portfolio
    benchmark_orders = pd.DataFrame(index=orders.index, columns=['Order', 'Symbol', 'Shares'])
    
    # Buy and hold position
    benchmark_orders.iloc[0] = ['BUY', 'JPM', 1000]
    benchmark_orders.iloc[1:] = [['HOLD', 'JPM', 0] for date in benchmark_orders.index[1:]]
    
    print('benchmark_orders\n', benchmark_orders)
    
    benchmark_portvals = mm.compute_portvals(benchmark_orders, start_val=sv)
    benchmark_portvals = benchmark_portvals / benchmark_portvals.iloc[0]

    # Calculate metrics for portfolio
    daily_returns_portfolio = portvals.pct_change().dropna()
    cum_return_portfolio = (portvals.iloc[-1] / portvals.iloc[0]) - 1
    std_daily_return_portfolio = daily_returns_portfolio.std()
    avg_daily_return_portfolio = daily_returns_portfolio.mean()

    # Calculate metrics for benchmark
    daily_returns_benchmark = benchmark_portvals.pct_change().dropna()
    cum_return_benchmark = (benchmark_portvals.iloc[-1] / benchmark_portvals.iloc[0]) - 1
    std_daily_return_benchmark = daily_returns_benchmark.std()
    avg_daily_return_benchmark = daily_returns_benchmark.mean()

    # Print metrics
    print(f"Cumulative Return of Portfolio: {round(cum_return_portfolio.iat[0], 8)}")
    print(f"Standard Deviation of Portfolio: {round(std_daily_return_portfolio.iat[0], 8)}")
    print(f"Average Daily Return of Portfolio: {round(avg_daily_return_portfolio.iat[0], 8)}")

    print(f"Cumulative Return of Benchmark: {round(cum_return_benchmark.iat[0], 8)}")
    print(f"Standard Deviation of Benchmark: {round(std_daily_return_benchmark.iat[0], 8)}")
    print(f"Average Daily Return of Benchmark: {round(avg_daily_return_benchmark.iat[0], 8)}")


    # Plot portfolio values and benchmark
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(portvals.index, portvals['Portfolio Value'], label='Portfolio Value')
    ax.plot(benchmark_portvals.index, benchmark_portvals['Portfolio Value'], label='Benchmark', linestyle='--')
    ax.set_title('Portfolio Value Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Portfolio Value')
    ax.legend()
    plt.tight_layout()
    save_plot(fig, 'Portfolio_Value_Over_Time')
    plt.show()
    print('Portfolio Value Plot Saved')
        
    # plt.savefig('./images/Portfolio_Value_Over_Time.png')
    # save_plot(fig, 'Portfolio Value Over Time')

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

    # # Plot indicators
    # fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    # # plt.figure(figsize=(14, 10))

    # # Bollinger Bands
    # ax = axs[0, 0]
    # bb_df = indicators['bb_value']
    # ax.plot(prices.index, prices, label='Price')
    # ax.plot(prices.index, bb_df['rolling mean'], label='Rolling Mean', linestyle='-')
    # ax.plot(prices.index, bb_df['upper band'], label='Upper Band', linestyle='-')
    # ax.plot(prices.index, bb_df['lower band'], label='Lower Band', linestyle='-')
    # ax.plot(prices.index, bb_df['bb value'], label='BB Value', linestyle=':')
    # ax.set_title('Bollinger Bands')
    # ax.legend()

    # # Simple Moving Average
    # ax = axs[0, 1]
    # sma_df = simple_moving_average(prices)
    # ax.plot(prices.index, prices, label='Price')
    # ax.plot(prices.index, sma_df['simple moving average'], label='SMA', linestyle='-')
    # ax.set_title('Simple Moving Average')
    # ax.legend()

    # # Relative Strength Index
    # ax = axs[1, 0]
    # ax.plot(prices.index, prices, label='Price', linestyle='-', alpha=0.5)
    # ax.plot(prices.index, relative_strength_index(prices), label='RSI')
    # ax.set_title('Relative Strength Index')
    # ax.legend()

    # # Momentum
    # ax1 = axs[1, 1]
    # ax2 = ax1.twinx()
    # ax1.plot(prices.index, prices, label='Price', linestyle='-', alpha=0.5, color='b')
    # ax2.plot(prices.index, momentum(prices), label='Momentum', color='r')
    # ax1.set_title('Momentum')
    # ax1.set_ylabel('Price', color='b')
    # ax2.set_ylabel('Momentum', color='r')
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')

    # # Commodity Channel Index
    # ax = axs[2, 0]
    # cci_df = commodity_channel_index(prices)
    # ax.plot(prices.index, cci_df['Commodity Channel Index'], label='CCI')
    # ax.plot(prices.index, cci_df['Normalized Price'], label='Normalized Price', linestyle='-')
    # ax.set_title('Commodity Channel Index')
    # ax.legend()

    # plt.tight_layout()
    # save_plot(fig, f'Indicators_{symbol}')

    # Plot Bollinger Bands
    fig, ax = plt.subplots(figsize=(10, 6))
    bb_df = indicators['bb_value']
    ax.plot(prices.index, prices, label='Price')
    ax.plot(prices.index, bb_df['rolling mean'], label='Rolling Mean', linestyle='-')
    ax.plot(prices.index, bb_df['upper band'], label='Upper Band', linestyle='-')
    ax.plot(prices.index, bb_df['lower band'], label='Lower Band', linestyle='-')
    ax.plot(prices.index, bb_df['bb value'], label='BB Value', linestyle=':')
    ax.set_title('Bollinger Bands')
    ax.legend()
    plt.tight_layout()
    save_plot(fig, f'Bollinger_Bands_{symbol}')

    # Plot Simple Moving Average
    fig, ax = plt.subplots(figsize=(10, 6))
    sma_df = simple_moving_average(prices)
    ax.plot(prices.index, prices, label='Price')
    ax.plot(prices.index, sma_df['simple moving average'], label='SMA', linestyle='-')
    ax.set_title('Simple Moving Average')
    ax.legend()
    plt.tight_layout()
    save_plot(fig, f'Simple_Moving_Average_{symbol}')

    # Plot Relative Strength Index
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(prices.index, prices, label='Price', linestyle='-', alpha=0.5)
    ax.plot(prices.index, relative_strength_index(prices), label='RSI')
    ax.set_title('Relative Strength Index')
    ax.legend()
    plt.tight_layout()
    save_plot(fig, f'Relative_Strength_Index_{symbol}')

    # Plot Momentum
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(prices.index, prices, label='Price', linestyle='-', alpha=0.5, color='b')
    ax2.plot(prices.index, momentum(prices), label='Momentum', color='r')
    ax1.set_title('Momentum')
    ax1.set_ylabel('Price', color='b')
    ax2.set_ylabel('Momentum', color='r')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    save_plot(fig, f'Momentum_{symbol}')

    # Plot Commodity Channel Index
    fig, ax = plt.subplots(figsize=(10, 6))
    cci_df = commodity_channel_index(prices)
    ax.plot(prices.index, cci_df['Commodity Channel Index'], label='CCI')
    ax.plot(prices.index, cci_df['Normalized Price'], label='Normalized Price', linestyle='-')
    ax.set_title('Commodity Channel Index')
    ax.legend()
    plt.tight_layout()
    save_plot(fig, f'Commodity_Channel_Index_{symbol}')

    # Run the test policy
    run_test_policy()

if __name__ == "__main__":
    main()