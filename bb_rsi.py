import config
import logging
import asyncio
import requests
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.utils import dropna
from alpaca.data.historical import HistoricalDataClient
from alpaca.common.time import TimeFrame
import json
import plotly.graph_objects as go
import plotly.express as px

# import talib as ta
# import plotly.graph_objects as go
# import plotly.express as px

# ENABLE LOGGING - options, DEBUG,INFO, WARNING?
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Alpaca API
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'
ALPACA_DATA_URL = 'https://data.alpaca.markets'

HEADERS = {'APCA-API-KEY-ID': config.APCA_API_KEY_ID,
           'APCA-API-SECRET-KEY': config.APCA_API_SECRET_KEY}


# Alpaca Rest client
client = HistoricalDataClient(
    config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY)

# Trading variables
trading_pair = 'BTCUSD'
exchange = 'FTXU'
start_date = '2020-01-01'
today = date.today()
today = today.strftime("%Y-%m-%d")
rsi_upper_bound = 60
rsi_lower_bound = 40

bar_data = 0
latest_bar_data = 0

# Wait time between each bar request -> 1 hour
waitTime = 3600


async def main():
    '''
    Get historical data from Alpaca and calculate RSI and Bollinger Bands.
    Backtest historical data to determine buy/sell/hold decisions and test performance.
    After backtesting, plot the results. Then, enter the loop to wait for new data and
    calculate entry and exit decisions.
    '''
    # Log the current balance of the MATIC token in our Alpaca account
    logger.info('BTC Position on Alpaca: {0}'.format(get_positions()))
    # Log the current Cash Balance (USD) in our Alpaca account
    logger.info("USD position on Alpaca: {0}".format(
        get_account_details()['cash']))
    get_crypto_bar_data(trading_pair, start_date, today, exchange)

    # Plot Bollinger bands from start date to today
    plot_signals()
    while True:
        l1 = loop.create_task(get_crypto_bar_data(
            trading_pair, start_date, today, exchange))
        # Wait for the tasks to finish
        await asyncio.wait([l1])
        await check_condition()
        # Wait for the a certain amount of time between each quote request
        await asyncio.sleep(waitTime)


async def get_crypto_bar_data(trading_pair, start_date, end_date, exchange):
    '''
    Get bar data from Alpaca for a given trading pair and exchange
    '''
    print("ENd date is: ", end_date)
    try:

        bars = client.get_crypto_bars(
            trading_pair, TimeFrame.Hour, start=start_date, end=end_date, limit=10000, exchanges=exchange)

        bars = bars.json()

        bars = json.loads(bars)

        bars = bars['bar_set'][trading_pair]

        bars = pd.DataFrame(bars)
        bars = bars.drop(
            columns=["open", "high", "low", "trade_count", "symbol", "timeframe", "exchange"], axis=1)

        bars = get_rsi(bars)

        bars = get_bb(bars)
        bars = bars.dropna()
        bars = bars.reset_index(drop=True)

        # Assigning bar data to global variables
        global latest_bar_data
        global bar_data
        bar_data = bars
        latest_bar_data = bars[-1:]
    # If there is an error, log it
    except Exception as e:
        logger.exception(
            "There was an issue getting trade quote from Alpaca: {0}".format(e))
        return False

    return bars


async def check_condition():
    logger.info("Checking Buy/Sell conditions for Bollinger bands and RSI")
    if latest_bar_data.empty:
        logger.info("Unable to get latest bar data")
    # If bollinger high indicator is 1 and RSI is above the upperbound, then buy
    if latest_bar_data['bb_hi'] == 1 and latest_bar_data['rsi'] > rsi_upper_bound:
        logger.info(
            "Sell signal: Bollinger bands and RSI are above upper bound")
    elif latest_bar_data['bb_li'] == 1 and latest_bar_data['rsi'] < rsi_lower_bound:
        logger.info("Buy signal: Bollinger bands and RSI are below lower bound")
    else:
        logger.info("Hold signal: Bollinger bands and RSI are within bounds")


# bars = get_crypto_bar_data('BTCUSD', '2022-01-01', today, 'FTXU')


def get_daily_returns(df):
    df['daily_returns'] = df['close'].pct_change()
    return df


bars = get_daily_returns(bars)


def get_cumulative_return(df):
    df['cumulative_return'] = bars['daily_returns'].add(1).cumprod().sub(1)
    return df


bars = get_cumulative_return(bars)


def get_buy_hold_returns(df):
    df['buy_hold_returns'] = (df['cumulative_return'] + 1) * 10000
    return df


bars = get_buy_hold_returns(bars)


# forward fill any missing data points in our buy & hold strategies
# and forward fill BTC_daily_return for missing data points
bars[['buy_hold_returns', 'daily_returns', ]] = bars[[
    'buy_hold_returns', 'daily_returns']].ffill()


print(bars)


def get_bb(df):
    # calculate bollinger bands
    indicator_bb = BollingerBands(
        close=df["close"], window=20, window_dev=2)
    df['bb_mavg'] = indicator_bb.bollinger_mavg()
    df['bb_high'] = indicator_bb.bollinger_hband()
    df['bb_low'] = indicator_bb.bollinger_lband()

    # Add Bollinger Band high indicator
    df['bb_hi'] = indicator_bb.bollinger_hband_indicator()
    # Add Bollinger Band low indicator
    df['bb_li'] = indicator_bb.bollinger_lband_indicator()
    # Add Width Size Bollinger Bands
    df['bb_w'] = indicator_bb.bollinger_wband()
    # Add Percentage Bollinger Bands
    df['bb_p'] = indicator_bb.bollinger_pband()
    # print(df)
    return df


def get_rsi(df):
    indicator_rsi = RSIIndicator(close=df["close"], window=14)
    df['rsi'] = indicator_rsi.rsi()
    return df


bars = get_rsi(bars)

bars = get_bb(bars)
print(bars.shape)
bars = bars.dropna()
print(bars.shape)
print(bars.loc[bars['rsi'] > rsi_upper_bound].shape)
print(bars.loc[bars['bb_hi'] > 0].shape)
bars = bars.reset_index(drop=True)
print(bars[-1:])


def sell_points(df):
    return bars.loc[(bars['rsi'] > rsi_upper_bound) & (bars['bb_hi'] > 0) & (bars['rsi'].shift() < rsi_upper_bound) & (bars['bb_hi'].shift() == 0)]


def buy_points(df):
    return bars.loc[(bars['rsi'] < rsi_lower_bound) & (bars['bb_li'] > 0) & (bars['rsi'].shift() > rsi_lower_bound) & (bars['bb_li'].shift() == 0)]


# print("RSI IS >70 here:\n", bars.loc[bars['rsi'] > 70])
# print("BB bands are overbought here:\n", bars.loc[bars['bb_hi'] > 0])
buying_points = buy_points(bars)
selling_points = sell_points(bars)
print("Buying Points are: \n", buying_points)
print("Selling Points are: \n", selling_points)


buying_points['order'] = 'buy'
selling_points['order'] = 'sell'

# Combine buys and sells into 1 data frame
orders = pd.concat([buying_points[['order']],
                   selling_points[['order']]]).sort_index()

# new dataframe with market data and orders merged
portfolio = pd.merge(bars, orders, how='outer',
                     left_index=True, right_index=True)


def backtest_returns(df):

    # Backtest of SMA crossover strategy
    active_position = False
    equity = 10000

    # Iterate row by row of our historical data
    for index, row in df.iterrows():

        # change state of position
        if row['order'] == 'buy':
            active_position = True
        elif row['order'] == 'sell':
            active_position = False

        # update strategy equity
        if active_position:
            df.loc[index, 'trading_returns'] = (
                row['daily_returns'] + 1) * equity
            equity = df.loc[index, 'trading_returns']
        else:
            df.loc[index, 'trading_returns'] = equity
        fig.add_trace(go.Scatter(x=portfolio['timestamp'], y=portfolio['close'],
                      name='Buy', mode='markers', marker_color='green', marker_symbol=[49], marker_size=10))
        fig.add_trace(go.Scatter(x=selling_points['timestamp'], y=selling_points['close'],
                      name='Sell', mode='markers', marker_color='red', marker_symbol=[50], marker_size=10))

    fig = px.line(portfolio[['trading_returns', 'buy_hold_returns']],
                  color_discrete_sequence=['green', 'blue'])
    fig.show()

    return df


# portfolio = backtest_returns(portfolio)

print("Portfolio after backtesting: \n", portfolio)


def get_positions():
    '''
    Get positions on Alpaca
    '''
    try:
        positions = requests.get(
            '{0}/v2/positions'.format(ALPACA_BASE_URL), headers=HEADERS)
        # logger.info('Alpaca positions reply status code: {0}'.format(
        # positions.status_code))
        if positions.status_code != 200:
            logger.info(
                "Undesirable response from Alpaca! {}".format(positions.json()))
            return False
        # positions = positions[0]
        matic_position = positions.json()[0]['qty']
        # logger.info('Matic Position on Alpaca: {0}'.format(matic_position))
    except Exception as e:
        logger.exception(
            "There was an issue getting positions from Alpaca: {0}".format(e))
        return False
    return matic_position


def get_account_details():
    '''
    Get Alpaca Trading Account Details
    '''
    try:
        account = requests.get(
            '{0}/v2/account'.format(ALPACA_BASE_URL), headers=HEADERS)
        if account.status_code != 200:
            logger.info(
                "Undesirable response from Alpaca! {}".format(account.json()))
            return False
    except Exception as e:
        logger.exception(
            "There was an issue getting account details from Alpaca: {0}".format(e))
        return False
    return account.json()


# Post an Order to Alpaca
def post_Alpaca_order(symbol, qty, side, type, time_in_force):
    '''
    Post an order to Alpaca
    '''
    try:
        order = requests.post(
            '{0}/v2/orders'.format(BASE_ALPACA_URL), headers=HEADERS, json={
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'type': type,
                'time_in_force': time_in_force,
            })
        logger.info('Alpaca order reply status code: {0}'.format(
            order.status_code))
        if order.status_code != 200:
            logger.info(
                "Undesirable response from Alpaca! {}".format(order.json()))
            return False
    except Exception as e:
        logger.exception(
            "There was an issue posting order to Alpaca: {0}".format(e))
        return False
    return order.json()


def plot_signals():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bars['timestamp'],
                  y=bars['close'], name='Closing Price', fill=None, mode='lines', line_color='black'))
    fig.add_trace(go.Scatter(x=bars['timestamp'],
                  y=bars['bb_mavg'], name='Moving Average', fill=None, mode='lines', line_color='blue'))
    fig.add_trace(go.Scatter(x=bars['timestamp'],
                  y=bars['bb_high'], name='BB_High', fill='tonexty'))
    fig.add_trace(go.Scatter(x=bars['timestamp'],
                  y=bars['bb_low'], name='BB_Low', fill='tonexty'))
    # print(buying_points.timestamp)
    fig.add_trace(go.Scatter(
        x=buying_points['timestamp'], y=buying_points['close'], name='Buy', mode='markers', marker_color='green', marker_symbol=[49], marker_size=10))
    fig.add_trace(go.Scatter(x=selling_points['timestamp'], y=selling_points['close'],
                  name='Sell', mode='markers', marker_color='red', marker_symbol=[50], marker_size=10))
    fig.show()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
