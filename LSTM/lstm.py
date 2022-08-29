from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoQuotesRequest, CryptoTradesRequest
from alpaca.trading.requests import GetOrdersRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import json
import logging
import config
import asyncio


# ENABLE LOGGING - options, DEBUG,INFO, WARNING?
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Alpaca Trading Client
trading_client = TradingClient(
    config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY, paper=True)

# Alpaca Market Data Client
data_client = CryptoHistoricalDataClient()

# Trading variables
trading_pair = 'ETH/USD'
# Wait time between each bar request
waitTime = 60
data = 0


async def main():
    '''
    Main function to get latest asset data and check possible trade conditions
    '''

    # closes all position AND also cancels all open orders
    # trading_client.close_all_positions(cancel_orders=True)
    # logger.info("Closed all positions")

    while True:
        logger.info('----------------------------------------------------')
        l1 = loop.create_task(get_crypto_bar_data(
            trading_pair))
        # Wait for the tasks to finish
        await asyncio.wait([l1])
        # Check if any trading condition is met
        # await check_condition()
        # Wait for the a certain amount of time between each bar request
        await asyncio.sleep(waitTime)


async def get_crypto_bar_data(trading_pair):
    '''
    Get Crypto Bar Data from Alpaca for the last 1000 hours
    '''
    time_diff = datetime.now() - relativedelta(hours=1000)
    logger.info("Getting crypto bar data for {0} from {1}".format(
        trading_pair, time_diff))
    # Defining Bar data request parameters
    request_params = CryptoBarsRequest(
        symbol_or_symbols=[trading_pair],
        timeframe=TimeFrame.Hour,
        start=time_diff
    )
    # Get the bar data from Alpaca
    bars_df = data_client.get_crypto_bars(request_params).df

    bars_df = bars_df.reset_index()
    bars_df.set_index('timestamp', inplace=True)
    global data
    # data = bars_df

    # data = data.set_index(data['timestamp'])
    # print("works till here")
    # data = data.set_index('timestamp')
    # print("hello")
    # print(data)
    aim = 'close'
    data = bars_df.filter([aim])
    # data = data.values
    print(data)
    # data = data.values
    # print(data)
    train_data = data.iloc[:800]
    test_data = data.iloc[800:]

    print(train_data)
    print(test_data)
    line_plot(train_data[aim], test_data[aim],
              'training', 'test', title='')
    # print("here")
    # Calculate the order prices
    # global buying_price, selling_price, current_position
    # buying_price, selling_price = calc_order_prices(bars_df)

    # if len(get_positions()) > 0:

    #     current_position = float(json.loads(get_positions()[0].json())['qty'])

    #     buy_order = False
    # else:
    #     sell_order = False
    return bars_df


def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('ETHUSD', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    plt.savefig('line_plot.png')


def build_lstm_model(input_data, output_size, neurons, activ_func='linear',
                     dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(
        input_data.shape[1], input_data.shape[2]), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
