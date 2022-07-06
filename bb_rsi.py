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
# today = pd.Timestamp.today()
print(today)


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
    get_crypto_bar_data(trading_pair, start_date, exchange)

    while True:
        l1 = loop.create_task(get_crypto_bar_data(
            trading_pair, start_date, exchange))
        # Wait for the tasks to finish
        await asyncio.wait([l1])
        await check_arbitrage()
        # Wait for the a certain amount of time between each quote request
        await asyncio.sleep(waitTime)


def get_crypto_bar_data(trading_pair, start_date, end_date, exchange):
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
        # print(bars)
    # If there is an error, log it
    except Exception as e:
        logger.exception(
            "There was an issue getting trade quote from Alpaca: {0}".format(e))
        return False

    return bars


bars = get_crypto_bar_data('BTCUSD', '2022-01-01', today, 'FTXU')


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
print(bars.loc[bars['rsi'] > 70].shape)
print(bars.loc[bars['bb_hi'] > 0].shape)
bars = bars.reset_index(drop=True)
print(bars[-1:])


def add_condition(df):
    for i in range(len(df)):
        # Check if last bar is above 70% RSI and above Bollinger Band high then sell
        if df.loc[i, 'rsi'] > 70 and df.loc[i, 'bb_hi'] > 0:
            df.loc[i, 'opinion'] = 'Sell'
        # Check if last bar is below 30% RSI and below Bollinger Band low then buy
        elif((df.loc[i, 'rsi'] < 30) & (df.loc[i, 'bb_li'] > 0)):
            df.loc[i, 'opinion'] = 'Buy'
        else:
            df.loc[i, 'opinion'] = 'Hold'

    return bars


bars = add_condition(bars)
print(add_condition(bars))


def buy_points(df):
    return bars.loc[(bars['rsi'] > 70) & (bars['bb_hi'] > 0) & (bars['rsi'].shift() < 70) & (bars['bb_hi'].shift() == 0)]


def sell_points(df):
    return bars.loc[(bars['rsi'] < 30) & (bars['bb_li'] > 0) & (bars['rsi'].shift() > 30) & (bars['bb_li'].shift() == 0)]


print("RSI IS >70 here:\n", bars.loc[bars['rsi'] > 70])
print("BB bands are overbought here:\n", bars.loc[bars['bb_hi'] > 0])
buying_points = buy_points(bars)
selling_points = sell_points(bars)
print(buying_points)
print(selling_points)


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
    # calculating when Buy appears after a Hold
    buy_signal = bars[(bars['opinion'] == 'Buy') &
                      (bars['opinion'].shift() == 'Hold')]

    # calculating when Sell appears after a Hold
    sell_signal = bars[(bars['opinion'] == 'Sell') &
                       (bars['opinion'].shift() == 'Hold')]
    # Plot green upward facing triangles at crossovers
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bars['timestamp'], y=bars['bb_low']))
    fig.add_trace(px.scatter(buy_signal, x=bars['timestamp'], y=bars['bb_low'],
                             color_discrete_sequence=['green'], symbol_sequence=[49]))
    # fig = px.scatter(buy_signal, x=bars['timestamp'], y=bars['bb_low'],
    #                  color_discrete_sequence=['green'], symbol_sequence=[49])

    # Plot red downward facing triangles at crossunders
    fig.add_trace(go.Scatter(x=bars['timestamp'], y=bars['bb_high']))
    fig.add_trace(px.scatter(sell_signal, x=bars['timestamp'], y=bars['bb_high'], color_discrete_sequence=[
        'red'], symbol_sequence=[50]))
    # fig = px.scatter(sell_signal, x=bars['timestamp'], y=bars['bb_high'], color_discrete_sequence=[
    #     'red'], symbol_sequence=[50])
    # Plot slow sma, fast sma and price
    fig.add_trace(go.Scatter(x=bars['timestamp'], y=bars['close']))
    fig.add_trace(go.Scatter(x=bars['timestamp'], y=bars['bb_mavg']))
    fig.add_trace(go.Scatter(x=bars['timestamp'], y=bars['bb_high']))
    fig.add_trace(go.Scatter(x=bars['timestamp'], y=bars['bb_low']))
    fig.show()


# plot_signals()
# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())
# loop.close()
