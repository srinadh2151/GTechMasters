"""
CS7646 ML For Trading
Project 6: Indicator Evaluation
Indicators library
Srinadh Nidadana (snidadana3)

This file provides technical indicators for use in the Manual Strategy function.
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import util as ut

def bollinger_bands(prices, window=10):

    """Calculate Bollinger Bands."""
    rm = prices.rolling(window=window, center=False).mean()
    sd = prices.rolling(window=window, center=False).std()
    upband = rm + (2 * sd)
    dnband = rm - (2 * sd)
    
    # Create a DataFrame to store the indicators
    df_indicators = pd.DataFrame(index=prices.index)
    df_indicators['rolling mean'] = rm
    df_indicators['upper band'] = upband
    df_indicators['lower band'] = dnband
    
    # Calculate BB value
    bb_value = (prices - rm) / (2 * sd)
    df_indicators['bb value'] = bb_value
    
    return df_indicators

def simple_moving_average(prices, window=20):
    """Calculate Simple Moving Average."""
    sma = prices.rolling(window=window).mean()

    df_indicators = pd.DataFrame(index=prices.index)
    df_indicators['simple moving average'] = sma
    # df_indicators['price_sma_ratio'] = prices / sma
    
    return df_indicators 

def relative_strength_index(prices, window=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def momentum(prices, window=10):
    """Calculate Momentum."""
    momentum = (prices / prices.shift(window)) - 1
    return momentum

def commodity_channel_index(prices, window=20):
    
    """Calculate Commodity Channel Index."""
    rm = prices.rolling(window=window).mean()
    normalized_prices = (prices - prices.min()) / (prices.max() - prices.min())

    cci = (prices - rm) / (2.5 * prices.rolling(window=window).std())
    
    df_indicators = pd.DataFrame(index=prices.index)
    df_indicators['Commodity Channel Index'] = cci
    df_indicators['Normalized Price'] = normalized_prices

    return df_indicators

def author():
    return 'snidadana3'

def test_code():
    # Define date range
    dev_sd = dt.datetime(2008, 1, 1)
    dev_ed = dt.datetime(2009, 12, 31)
    symbol = 'JPM'
    
    # Get stock data
    dates = pd.date_range(dev_sd, dev_ed)
    prices_all = ut.get_data([symbol], dates)
    prices = prices_all[symbol]
    
    # Plot indicators
    plt.figure(figsize=(14, 10))
    
    # Bollinger Bands
    plt.subplot(3, 2, 1)
    bb_df = bollinger_bands(prices)
    plt.plot(prices.index, prices, label='Price')
    plt.plot(prices.index, bb_df['rolling mean'], label='Rolling Mean', linestyle='-')
    plt.plot(prices.index, bb_df['upper band'], label='Upper Band', linestyle='-')
    plt.plot(prices.index, bb_df['lower band'], label='Lower Band', linestyle='-')
    plt.plot(prices.index, bb_df['bb value'], label='BB Value', linestyle=':')
    
    plt.title('Bollinger Bands')
    plt.legend()
    
    # Simple Moving Average
    plt.subplot(3, 2, 2)
    sma_df = simple_moving_average(prices)
    plt.plot(prices.index, prices, label='Price')
    plt.plot(prices.index, sma_df['simple moving average'], label='SMA', linestyle='-')
    # plt.plot(prices.index, sma_df['price_sma_ratio'], label='Price/SMA Ratio', linestyle=':')
    plt.title('Simple Moving Average')
    plt.legend()
    
    # Relative Strength Index
    # plt.subplot(3, 2, 3)
    # plt.plot(prices.index, relative_strength_index(prices), label='RSI')
    # plt.title('Relative Strength Index')
    # plt.legend()
    plt.subplot(3, 2, 3)
    plt.plot(prices.index, prices, label='Price', linestyle='-', alpha=0.5)
    plt.plot(prices.index, relative_strength_index(prices), label='RSI')
    plt.title('Relative Strength Index')
    plt.legend()
    
    # # Momentum
    # plt.subplot(3, 2, 4)
    # plt.plot(prices.index, prices, label='Price', linestyle='-', alpha=0.5)
    # plt.plot(prices.index, momentum(prices), label='Momentum')
    # plt.title('Momentum')
    # plt.legend()

    # Momentum
    plt.subplot(3, 2, 4)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax1.plot(prices.index, prices, label='Price', linestyle='-', alpha=0.5, color='b')
    ax2.plot(prices.index, momentum(prices), label='Momentum', color='r')

    ax1.set_title('Momentum')
    ax1.set_ylabel('Price', color='b')
    ax2.set_ylabel('Momentum', color='r')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Commodity Channel Index
    plt.subplot(3, 2, 5)
    cci_df = commodity_channel_index(prices)
    plt.plot(prices.index, cci_df['Commodity Channel Index'], label='CCI')
    plt.plot(prices.index, cci_df['Normalized Price'], label='Normalized Price', linestyle='-')
    plt.title('Commodity Channel Index')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_code()