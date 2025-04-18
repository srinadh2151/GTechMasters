"""
Custom Implementation of a Manual Trading Strategy
"""
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from indicators import bollinger_bands, momentum, simple_moving_average, relative_strength_index
from marketsimcode import compute_portvals
from util import get_data
import os

# os.chdir('./CS7646 ML4T/ML4T_2025Spring/strategy_evaluation')

class ManualStrategy():
    """
    Custom implementation of a manual trading strategy using multiple indicators.
    """

    def __init__(self):
        # To Keep track of our LONG and SHORT entry points
        self.entry_points = []

    def get_entry_points(self):
        return self.entry_points

    def testPolicy(self, symbol="AAPL", start_date=dt.datetime(2010, 1, 1), end_date=dt.datetime(2011, 12, 31), start_value=100000):
        """
        Determines the trading strategy based on indicators.

        Args:
            symbol (str): Stock symbol.
            start_date (datetime): Start date for trading.
            end_date (datetime): End date for trading.
            start_value (int): Starting portfolio value.

        Returns:
            pd.DataFrame: DataFrame of orders with columns: Order, Date, Symbol, Shares.
        """
        dates = pd.date_range(start_date, end_date)
        prices, highs, lows, volumes = self._fetch_market_data(symbol, dates)
        indicators = self._compute_indicators(prices, highs, lows, volumes, lookback=10)
        return self._generate_trading_orders(symbol, prices.index, indicators)

    @staticmethod
    def _fetch_market_data(symbol, dates):
        return (
            get_data([symbol], dates, colname='Adj Close').fillna(method='ffill').fillna(method='bfill'),
            get_data([symbol], dates, colname='High').fillna(method='ffill').fillna(method='bfill'),
            get_data([symbol], dates, colname='Low').fillna(method='ffill').fillna(method='bfill'),
            get_data([symbol], dates, colname='Volume').fillna(method='ffill').fillna(method='bfill')
        )

    @staticmethod
    def _compute_indicators(prices, highs, lows, volumes, lookback):
        mtm = momentum(prices, lookback)
        sma, sma_ratio = simple_moving_average(prices, lookback)
        bbands = bollinger_bands(prices, sma, lookback)
        rsi = relative_strength_index(prices, lookback)
        return mtm, sma_ratio, bbands, rsi

    def _generate_trading_orders(self, symbol, trading_dates, indicators):
        orders = pd.DataFrame(index=trading_dates, columns=['Order', 'Date', 'Symbol', 'Shares'])
        current_shares = 0

        for index, date in enumerate(trading_dates):
            yesterday = trading_dates[index - 1] if index > 0 else date

            if self._should_enter_long(symbol, date, yesterday, indicators, current_shares):
                order, shares_traded = self._enter_long(date, symbol, current_shares)
                self.entry_points.append((date, 'LONG'))
            elif self._should_enter_short(symbol, date, yesterday, indicators, current_shares):
                order, shares_traded = self._enter_short(date, symbol, current_shares)
                self.entry_points.append((date, 'SHORT'))
            elif self._should_exit_long(symbol, date, yesterday, indicators, current_shares):
                order, shares_traded = self._exit_long(date, symbol)
            elif self._should_exit_short(symbol, date, yesterday, indicators, current_shares):
                order, shares_traded = self._exit_short(date, symbol)
            else:
                order, shares_traded = self._hold_position(date, symbol)

            orders.loc[date] = order
            current_shares += shares_traded

        return orders
    
    @staticmethod
    def _should_enter_long(symbol, today, yesterday, indicators, current_shares):
        _, _, bbands, rsi = indicators
        return rsi.loc[today, symbol] < 30 and bbands.loc[today, symbol] < -1. and current_shares != 1000

    @staticmethod
    def _should_enter_short(symbol, today, yesterday, indicators, current_shares):
        _, _, bbands, rsi = indicators
        return rsi.loc[today, symbol] > 70 and bbands.loc[today, symbol] > 1 and current_shares != -1000

    @staticmethod
    def _should_exit_long(symbol, today, yesterday, indicators, current_shares):
        mtm, sma, _, _ = indicators

        sma_trending_down = sma.loc[yesterday, symbol] > 1 and sma.loc[today, symbol] <= 1

        mtm_today = mtm.loc[today, symbol]
        mtm_yesterday = mtm.loc[yesterday, symbol]
        drop_in_momentum = (mtm_today < mtm_yesterday) and ((mtm_yesterday - mtm_today) / mtm_yesterday) > 0.50

        return (sma_trending_down or drop_in_momentum) and current_shares == 1000

    @staticmethod
    def _should_exit_short(symbol, today, yesterday, indicators, current_shares):

        mtm, sma, _, _ = indicators

        sma_trending_up = sma.loc[yesterday, symbol] < 1 and sma.loc[today, symbol] >= 1

        mtm_today = mtm.loc[today, symbol]
        mtm_yesterday = mtm.loc[yesterday, symbol]
        increase_in_momentum = mtm_today > mtm_yesterday and (mtm_today - mtm_yesterday) / mtm_yesterday > 0.50

        return (sma_trending_up or increase_in_momentum) and current_shares == -1000

    @staticmethod
    def _enter_long(date, symbol, current_shares):
        to_buy = 1000       
        if current_shares == -1000:
            to_buy = 2000
        return ['BUY', date, symbol, to_buy], to_buy

    @staticmethod
    def _enter_short(date, symbol, current_shares):
        to_sell = 1000
        if current_shares == 1000:
            to_sell = 2000
        return ['SELL', date, symbol, to_sell], -to_sell

    @staticmethod
    def _exit_long(date, symbol):
        return ['SELL', date, symbol, 1000], -1000

    @staticmethod
    def _exit_short(date, symbol):
        return ['BUY', date, symbol, 1000], 1000

    @staticmethod
    def _hold_position(date, symbol):
        return ['HOLD', date, symbol, 0], 0

def main():
    manual_port_vals_in, manual_orders_in, entry_points_in = evaluate_manual_strategy(
        symbol="JPM",
        start_date=dt.datetime(2008, 1, 1),
        end_date=dt.datetime(2009, 12, 31),
        start_value=100000
    )
    benchmark_port_vals_in = evaluate_benchmark(manual_port_vals_in.index)

    manual_port_vals_out, manual_orders_out, entry_points_out = evaluate_manual_strategy(
        symbol="JPM",
        start_date=dt.datetime(2010, 1, 1),
        end_date=dt.datetime(2011, 12, 31),
        start_value=100000
    )
    benchmark_port_vals_out = evaluate_benchmark(manual_port_vals_out.index)

    manual_orders_in.to_csv('images/manual_strategy_orders_in_sample.csv')
    manual_orders_out.to_csv('images/manual_strategy_orders_out_sample.csv')

    manual_port_vals_in /= manual_port_vals_in.iloc[0]
    benchmark_port_vals_in /= benchmark_port_vals_in.iloc[0]
    manual_port_vals_out /= manual_port_vals_out.iloc[0]
    benchmark_port_vals_out /= benchmark_port_vals_out.iloc[0]

    plot_strategy(
        [
            (manual_port_vals_in.index, benchmark_port_vals_in),
            (manual_port_vals_in.index, manual_port_vals_in)
        ],
        entry_points_in,
        'Manual Strategy vs Benchmark - In Sample',
        'Date',
        'Normalized Value',
        colors=['purple', 'red'],
        legend=['Benchmark', 'Manual Strategy']
    )

    plot_strategy(
        [
            (manual_port_vals_out.index, benchmark_port_vals_out),
            (manual_port_vals_out.index, manual_port_vals_out)
        ],
        entry_points_out,
        'Manual Strategy vs Benchmark - Out of Sample',
        'Date',
        'Normalized Value',
        colors=['purple', 'red'],
        legend=['Benchmark', 'Manual Strategy']
    )

def evaluate_manual_strategy(symbol, start_date, end_date, start_value):
    strategy = ManualStrategy()
    orders = strategy.testPolicy(symbol=symbol, start_date=start_date, end_date=end_date, start_value=start_value)
    port_vals = compute_portvals(orders, start_val=start_value, commission=9.95, impact=0.005)
    cumulative_return, std_dr, mean_dr = get_portfolio_stats(port_vals)

    print(f"\n===== Manual Strategy Stats ({start_date.date()} to {end_date.date()}) =====")
    print("Final portfolio value: ", port_vals.iloc[-1])
    print("Cumulative return: ", cumulative_return)
    print("Std of daily returns: ", std_dr)
    print("Mean of daily returns: ", mean_dr)

    return port_vals, orders, strategy.get_entry_points()

def evaluate_benchmark(trading_dates):
    orders = pd.DataFrame(index=trading_dates, columns=['Order', 'Date', 'Symbol', 'Shares'])
    orders.iloc[0] = ['BUY', trading_dates[0], 'JPM', 1000]
    orders.iloc[1:] = [['HOLD', date, 'JPM', 0] for date in trading_dates[1:]]

    port_vals = compute_portvals(orders,
                                 start_val=100000,
                                 commission=9.95,
                                 impact=0.005
                                 )
    
    cumulative_return, std_dr, mean_dr = get_portfolio_stats(port_vals)

    print("\n===== Benchmark Strategy Stats =====")
    print("Final portfolio value: ", port_vals.iloc[-1])
    print("Cumulative return: ", cumulative_return)
    print("Std of daily returns: ", std_dr)
    print("Mean of daily returns: ", mean_dr)

    return port_vals

def get_portfolio_stats(portfolio_values):
    cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    daily_returns = (portfolio_values / portfolio_values.shift(1)) - 1
    std_dr = daily_returns.std()
    mean_dr = daily_returns.mean()
    return cumulative_return, std_dr, mean_dr

def plot_strategy(data, entry_points, title, xlabel, ylabel, colors=['purple', 'red'], legend=None):
    plt.close()
    fig, ax = plt.subplots()
    fig.autofmt_xdate()

    ax.grid(color='black', linestyle='dotted')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    for index, (xdata, ydata) in enumerate(data):
        plt.plot(xdata, ydata, linewidth=2.5, color=colors[index], 
                 alpha=0.4, label=legend[index])

    plt.title(title)
    plt.xlabel(xlabel, fontsize='15')
    plt.ylabel(ylabel, fontsize='15')

    if legend is not None:
        plt.legend(fontsize='small')

    for (date, entry_point) in entry_points:
        color = 'blue' if entry_point == 'LONG' else 'black'
        plt.axvline(x=date, color=color,
                    alpha=0.4, linewidth=2.0)

    plt.savefig(f"images/{title.replace(' ', '')}.png", bbox_inches='tight')

if __name__ == '__main__':
    main()