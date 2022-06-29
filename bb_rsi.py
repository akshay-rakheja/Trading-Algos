from sqlite3 import DatabaseError
import config
import logging
import asyncio
import requests
import pandas as pd

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

    # typical price
    TP = (df['h'] + df['l'] + df['c']) / 3
    # but we will use Adj close instead for now, depends

    data = TP
    #data = df['Adj Close']

    # takes one column from dataframe
    B_MA = pd.Series((data.rolling(n, min_periods=n).mean()), name='B_MA')
    sigma = data.rolling(n, min_periods=n).std()

    BU = pd.Series((B_MA + m * sigma), name='BU')
    BL = pd.Series((B_MA - m * sigma), name='BL')

    df = df.join(B_MA)
    df = df.join(BU)
    df = df.join(BL)

    return df


matic_bars = requests.get(
    '{0}/v1beta1/crypto/{1}/bars?timeframe={2}&start={3}'.format(ALPACA_DATA_URL, 'MATICUSD', '1Hour', '2022-01-01'), headers=HEADERS)

matic_bars = matic_bars.json()['bars']
matic_bars = pd.DataFrame(matic_bars)

bollinger_df = bollinger_bands(matic_bars, 20, 2)

print(matic_bars)
print(bollinger_df)
