import config
import logging
import asyncio
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta import add_all_ta_features
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


def get_alpaca_quote_data(trading_pair, exchange):
    '''
    Get trade quote data from Alpaca API
    '''
    # Try to get a quote from 1Inch
    try:
        # Get the current quote response for the trading pair (MATIC/USDC)
        quote = requests.get(
            '{0}/v1beta1/crypto/{1}/bars?timeframe={2}&start={3}'.format(ALPACA_DATA_URL, trading_pair, '15Min', '2022-01-01'), headers=HEADERS)

        # Status code 200 means the request was successful
        if quote.status_code != 200:
            logger.info(
                "Undesirable response from Alpaca! {}".format(quote.json()))
            return False

    # If there is an error, log it
    except Exception as e:
        logger.exception(
            "There was an issue getting trade quote from Alpaca: {0}".format(e))
        return False

    return quote.json()


def bollinger_bands(df, n, m):
    # takes dataframe on input
    # n = smoothing length
    # m = number of standard deviations away from MA

    # Using closing prices to calculate Bollinger Bands and MA
    data = df['c']
    B_MA = pd.Series((data.rolling(n, min_periods=n).mean()), name='B_MA')
    sigma = data.rolling(n, min_periods=n).std()

    # upper band
    BU = pd.Series((B_MA + m * sigma), name='BU')
    # lower band
    BL = pd.Series((B_MA - m * sigma), name='BL')

    # add to dataframe
    df = df.join(B_MA)
    df = df.join(BU)
    df = df.join(BL)
    df = df.dropna()
    print(df)
    df = df.reset_index(drop=True)
    print(df)
    return df


def add_signal(df):
    # adds two columns to dataframe with buy and sell signals
    buy_list = []
    sell_list = []
    print(df['h'][20])

    for i in range(len(df['c'])):
        # if df['Close'][i] > df['BU'][i]:           # sell signal     daily
        if df['h'][i] > df['BU'][i]:             # sell signal     weekly
            buy_list.append(np.nan)
            sell_list.append(df['c'][i])
        # elif df['Close'][i] < df['BL'][i]:         # buy signal      daily
        elif df['l'][i] < df['BL'][i]:            # buy signal      weekly
            buy_list.append(df['c'][i])
            sell_list.append(np.nan)
        else:
            buy_list.append(np.nan)
            sell_list.append(np.nan)

    buy_list = pd.Series(buy_list, name='Buy')
    sell_list = pd.Series(sell_list, name='Sell')

    df = df.join(buy_list)
    df = df.join(sell_list)

    return df


matic_bars = requests.get(
    '{0}/v1beta1/crypto/{1}/bars?timeframe={2}&start={3}'.format(ALPACA_DATA_URL, 'MATICUSD', '1Hour', '2022-01-01'), headers=HEADERS)

matic_bars = matic_bars.json()['bars']
matic_bars = pd.DataFrame(matic_bars)

bollinger_df = bollinger_bands(matic_bars, 20, 2)
matic_signal = add_signal(bollinger_df)


print(matic_bars)
print(bollinger_df)
print(matic_signal)
