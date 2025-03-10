import pandas as pd
import numpy as np
import datetime as dt
import util as ut

def author():
    return 'snidadana3'

def testPolicy_old(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    # Get stock prices
    dates = pd.date_range(sd, ed)
    prices_all = ut.get_data([symbol], dates)
    prices = prices_all[symbol]

    # Normalize the price series to start from 1
    normalized_prices = prices / prices.iloc[0]
    
    # # Initialize trades DataFrame
    # df_trades = pd.DataFrame(index=prices.index, columns=[symbol])
    # df_trades.values[:] = 0  # Start with no trades

    # # Iterate over prices to determine trades
    # for i in range(1, len(prices)):
    #     if prices[i] > prices[i - 1]:  # Price is going up
    #         df_trades.iloc[i] = 1000  # Buy
    #     elif prices[i] < prices[i - 1]:  # Price is going down
    #         df_trades.iloc[i] = -1000  # Sell

    # # Ensure holdings are within constraints
    # holdings = df_trades.cumsum()
    # df_trades[holdings > 1000] = 0
    # df_trades[holdings < -1000] = 0
    
    # Initialize DataFrames for storing order signals
    # order_signals = pd.DataFrame(index=normalized_prices.index)
    # final_orders = pd.DataFrame(index=normalized_prices.index)

    # # Generate buy/sell signals based on price movement
    # order_signals['Order'] = normalized_prices.diff().apply(lambda x: 'BUY' if x > 0 else 'SELL')

    # # Adjust signals to ensure proper order sequence
    # shifted_signals = order_signals['Order'].shift(1)
    # adjusted_signals = shifted_signals.apply(lambda x: 'SELL' if x == 'BUY' else 'BUY')
    # final_orders['Order'] = order_signals['Order'].combine_first(adjusted_signals).dropna()

    # # Add columns for symbol and number of shares
    # final_orders['Symbol'] = 'JPM'
    # final_orders['Shares'] = 1000

    # Ensure the DataFrame is sorted by date
    # final_orders.sort_index(inplace=True)
    # print('***final_orders\n', final_orders)
    df1 = pd.DataFrame(index=normalized_prices.index)
    df2 = pd.DataFrame(index=normalized_prices.index)
    
    df1['Order'] = normalized_prices < normalized_prices.shift(-1)
    df1['Order'].replace(True, 'BUY', inplace=True)
    df1['Order'].replace(False, 'SELL', inplace=True)

    # Create a DataFrame for the final orders
    # df2['ORDER'] = df1['ORDER'].append(
    #     df1['ORDER'].shift(1).replace('BUY', 'TMP').replace('SELL', 'BUY').replace('TMP', 'SELL').dropna()
    # )
    
    shifted_orders = df1['Order'].shift(1).replace('BUY', 'TMP').replace('SELL', 'BUY').replace('TMP', 'SELL').dropna()
    df2['Order'] = pd.concat([df1['Order'], shifted_orders]).drop_duplicates(keep='first')

    # Add additional columns for symbol and shares
    df2['Symbol'] = 'JPM'
    df2['Shares'] = 1000
    
    df2.sort_index(inplace=True)
    
    # # Initialize DataFrames for storing order signals
    # df1 = pd.DataFrame(index=prices.index)
    # df2 = pd.DataFrame(index=prices.index)
    
    # # Determine buy/sell signals based on price movement
    # df1['ORDER'] = prices < prices.shift(-1)
    # df1['ORDER'].replace(True, 'BUY', inplace=True)
    # df1['ORDER'].replace(False, 'SELL', inplace=True)

    # # Create a DataFrame for the final orders
    # shifted_orders = df1['ORDER'].shift(1).replace('BUY', 'TMP').replace('SELL', 'BUY').replace('TMP', 'SELL').dropna()
    # df2['ORDER'] = pd.concat([df1['ORDER'], shifted_orders]).drop_duplicates(keep='first')

    # # Add additional columns for symbol and shares
    # df2['SYMBOL'] = 'JPM'
    # df2['SHARES'] = 1000
    
    # df2.sort_index(inplace=True)
    # print('df2\n', df2[~df2['ORDER'].isnull()])
    
    print('df2\n', df2[~df2['Order'].isnull()])

    return df2

def get_prices(symbol, dates):
    prices = ut.get_data([symbol], dates)
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    return prices[symbol]

def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        """
        Computes the best possible trading strategy

        Args:
            - symbol: The stock symbol to use
            - sd: The start date
            - ed: The end date
            - sv: The starting value of the portfolio

        Returns:
            - A dataframe of orders of the form: Order | Date | Symbol | Shares
        """
        prices = get_prices(symbol, pd.date_range(sd, ed))
        tomorrows_prices = prices.shift(-1)
        orders = pd.DataFrame(index=prices.index, columns=['Order', 'Symbol', 'Shares'])

        # Determine buy/sell signals
        orders['Order'] = 'HOLD'
        orders.loc[tomorrows_prices > prices, 'Order'] = 'BUY'
        orders.loc[tomorrows_prices < prices, 'Order'] = 'SELL'

        # Add symbol and shares columns
        orders['Symbol'] = symbol
        orders['Shares'] = 1000

        # Adjust shares for switching positions
        orders.loc[(orders['Order'] == 'BUY') & (orders['Order'].shift(1) == 'SELL'), 'Shares'] = 2000
        orders.loc[(orders['Order'] == 'SELL') & (orders['Order'].shift(1) == 'BUY'), 'Shares'] = 2000

        # Remove 'HOLD' orders
        orders = orders[orders['Order'] != 'HOLD']
        
        print('Orders dataframe:\n', orders)

        return orders

if __name__ == "__main__":
    # Example usage
    df_trades = testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    # print(df_trades)