import config
import logging
import asyncio
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.utils import dropna
from alpaca.data.historical import HistoricalDataClient
from alpaca.common.time import TimeFrame
import json
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


def get_crypto_bar_data(trading_pair, start_date, exchange):
    '''
    Get bar data from Alpaca for a given trading pair and exchange
    '''
    # Try to get a quote from 1Inch
    try:

        bars = client.get_crypto_bars(
            trading_pair, TimeFrame.Hour, start=start_date, limit=10000, exchanges=exchange)

        bars = bars.json()

        bars = json.loads(bars)

        bars = bars['bar_set'][trading_pair]

        bars = pd.DataFrame(bars)
        bars = bars.drop(columns=["symbol", "timeframe", "exchange"], axis=1)
        # print(bars)
    # If there is an error, log it
    except Exception as e:
        logger.exception(
            "There was an issue getting trade quote from Alpaca: {0}".format(e))
        return False

    return bars


bars = get_crypto_bar_data('BTCUSD', '2022-01-01', 'FTXU')


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
bars = bars.dropna()
print(bars.loc[bars['rsi'] > 70])
print(bars.loc[bars['bb_hi'] > 0])

# print(bars.loc[bars['bb_hi'] > 0])


# def bollinger_bands(df, n, m):
#     # takes dataframe on input
#     # n = smoothing length
#     # m = number of standard deviations away from MA

#     # Using closing prices to calculate Bollinger Bands and MA
#     data = df['c']
#     B_MA = pd.Series((data.rolling(n, min_periods=n).mean()), name='B_MA')
#     sigma = data.rolling(n, min_periods=n).std()

#     # upper band
#     BU = pd.Series((B_MA + m * sigma), name='BU')
#     # lower band
#     BL = pd.Series((B_MA - m * sigma), name='BL')

#     # add to dataframe
#     df = df.join(B_MA)
#     df = df.join(BU)
#     df = df.join(BL)
#     df = df.dropna()
#     # print(df)
#     df = df.reset_index(drop=True)
#     # print(df)
#     return df


# def add_signal(df):
#     # adds two columns to dataframe with buy and sell signals
#     buy_list = []
#     sell_list = []
#     # print(df['high'][20])

#     for i in range(len(df['close'])):
#         # if df['Close'][i] > df['BU'][i]:           # sell signal     daily
#         if df['high'][i] > df['BU'][i]:             # sell signal     weekly
#             buy_list.append(np.nan)
#             sell_list.append(df['close'][i])
#         # elif df['Close'][i] < df['BL'][i]:         # buy signal      daily
#         elif df['low'][i] < df['BL'][i]:            # buy signal      weekly
#             buy_list.append(df['close'][i])
#             sell_list.append(np.nan)
#         else:
#             buy_list.append(np.nan)
#             sell_list.append(np.nan)

#     buy_list = pd.Series(buy_list, name='Buy')
#     sell_list = pd.Series(sell_list, name='Sell')

#     df = df.join(buy_list)
#     df = df.join(sell_list)

#     return df


# bars = add_signal(bars)

# print(bars)
# response = requests.get(
#     '{0}/v1beta1/crypto/{1}/bars?timeframe={2}&start={3}'.format(ALPACA_DATA_URL, 'MATICUSD', '1Hour', '2022-01-01'), headers=HEADERS)

# matic_bars = response.json()['bars']
# # print("Heres my response:     ", matic_bars)
# nxt_token = response.json()['next_page_token']
# matic_bars = pd.DataFrame(matic_bars)


# while (nxt_token):
#     print("Starting next page with token :", nxt_token)
#     nxtpg = requests.get('{0}/v1beta1/crypto/{1}/bars?timeframe={2}&start={3}?page_token={4}'.format(
#         ALPACA_DATA_URL, 'MATICUSD', '1Hour', '2022-01-01', nxt_token), headers=HEADERS)
#     print(nxtpg.json())
#     # nxtpg = nxtpg.json()['bars']
#     matic_bars = matic_bars.append(pd.DataFrame(nxtpg))
#     nxt_token = nxtpg.json()['next_page_token']
#     # matic_bars = matic_bars.append(pd.DataFrame(nxtpg_token.json()['bars']))

# print(matic_bars)


# Initialize Bollinger Bands Indicator
# indicator_bb = BollingerBands(close=matic_bars["c"], window=20, window_dev=2)

# # Add Bollinger Bands features
# matic_bars['bb_mavg'] = indicator_bb.bollinger_mavg()
# matic_bars['bb_high'] = indicator_bb.bollinger_hband()
# matic_bars['bb_low'] = indicator_bb.bollinger_lband()

# # Add Bollinger Band high indicator
# matic_bars['bb_hi'] = indicator_bb.bollinger_hband_indicator()

# # Add Bollinger Band low indicator
# matic_bars['bb_li'] = indicator_bb.bollinger_lband_indicator()

# # Add Width Size Bollinger Bands
# matic_bars['bb_w'] = indicator_bb.bollinger_wband()

# # Add Percentage Bollinger Bands
# matic_bars['bb_p'] = indicator_bb.bollinger_pband()
# print(matic_bars)
# matic_signal = add_signal(bollinger_df)
# bollinger_bands = ta.bo

# print(matic_bars)
# print(bollinger_df)
# print(matic_signal)
