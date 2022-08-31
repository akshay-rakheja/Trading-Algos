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
from sklearn.preprocessing import MinMaxScaler
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
waitTime = 600
data = 0

current_position, current_price = 0, 0


async def main():
    '''
    Main function to get latest asset data and check possible trade conditions
    '''

    # closes all position AND also cancels all open orders
    # trading_client.close_all_positions(cancel_orders=True)
    # logger.info("Closed all positions")

    while True:
        logger.info('----------------------------------------------------')

        pred = stockPred()

        val = pred.predictModel()
        print(val)
        # l1 = loop.create_task(get_crypto_bar_data(
        # trading_pair))
        # Wait for the tasks to finish
        # await asyncio.wait([l1])
        # global data
        # x_train, y_train, x_test, y_test = scale_data(data)
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
    aim = 'close'
    data = bars_df.filter([aim])
    print(data)
    train_data = data.iloc[:800]
    test_data = data.iloc[800:]

    print(train_data)
    print(test_data)
    line_plot(train_data[aim], test_data[aim],
              'training', 'test', title='')

    return bars_df


def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('ETHUSD', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    plt.savefig('line_plot.png')


class stockPred:
    def __init__(self,
                 past_days: int = 50,
                 trading_pair: str = 'BTCUSD',
                 exchange: str = 'FTXU',
                 feature: str = 'close',

                 look_back: int = 72,

                 neurons: int = 50,
                 activ_func: str = 'linear',
                 dropout: float = 0.2,
                 loss: str = 'mse',
                 optimizer: str = 'adam',
                 epochs: int = 20,
                 batch_size: int = 32,
                 output_size: int = 1,

                 retrain_freq: int = 24  # once a day
                 ):
        # self.history = datetime.timedelta(days=past_days)
        self.trading_pair = trading_pair
        self.exchange = exchange
        self.feature = feature

        self.look_back = look_back

        self.neurons = neurons
        self.activ_func = activ_func
        self.dropout = dropout
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_size = output_size

        self.retrain_freq = retrain_freq

    def getAllData(self):
        logger.info("Getting Data")
        # Alpaca Trading Client
        trading_client = TradingClient(
            config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY, paper=True)

        # Alpaca Market Data Client
        data_client = CryptoHistoricalDataClient()

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
        df = data_client.get_crypto_bars(request_params).df

        # bars_df = bars_df.reset_index()
        # bars_df.set_index('timestamp', inplace=True)

        # df = data_client.get_crypto_bars(self.trading_pair, TimeFrame.Hour,
        #  start=date.today() - self.history, end=date.today()).df
        return df

    def getFeature(self):
        df = self.getAllData()
        # df = df[df.exchange == self.exchange]
        logger.info("Getting feature data")
        data = df.filter([self.feature])
        data = data.values
        return data

    def scaleData(self):
        data = self.getFeature()
        logger.info("Scaling data")
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler

    # train on all data for which labels are available (train + test from dev)
    def getTrainData(self):
        scaled_data = self.scaleData()[0]
        logger.info("Getting training data")
        x, y = [], []
        for price in range(self.look_back, len(scaled_data)):
            x.append(scaled_data[price - self.look_back:price, :])
            y.append(scaled_data[price, :])

        return np.array(x), np.array(y)

    def LSTM_model(self, input_data):
        model = Sequential()
        model.add(LSTM(self.neurons, input_shape=(
            input_data.shape[1], input_data.shape[2]), return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.neurons, return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.neurons))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=self.output_size))
        model.add(Activation(self.activ_func))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def trainModel(self):

        x, y = self.getTrainData()
        x_train = x[: len(x) - 1]
        y_train = y[: len(x) - 1]
        model = self.LSTM_model(x_train)
        logger.info("Training model")
        modelfit = model.fit(x_train, y_train, epochs=self.epochs,
                             batch_size=self.batch_size, verbose=1, shuffle=True)
        return model, modelfit

    def predictModel(self):

        scaled_data, scaler = self.scaleData()
        x_pred = scaled_data[-self.look_back:]
        x_pred = np.reshape(x_pred, (1, x_pred.shape[0]))

        model = self.trainModel()[0]
        pred = model.predict(x_pred).squeeze()
        pred = np.array([float(pred)])
        pred = np.reshape(pred, (pred.shape[0], 1))

        logger.info("Predicting Value")
        pred_true = scaler.inverse_transform(pred)
        return pred_true


async def check_condition():
    '''
    Strategy:
    - If the predicted price an hour from now is above the current price and we do not have a position, buy
    - If the predicted price an hour from now is below the current price and we do have a position, sell

    '''
    global current_position, current_price
    logger.info("Current Position is: {0}".format(current_position))
    logger.info("Buy Order status: {0}".format(buy_order))
    logger.info("Sell Order status: {0}".format(sell_order))
    logger.info("Buy_order_price: {0}".format(buy_order_price))
    logger.info("Sell_order_price: {0}".format(sell_order_price))
    # If the spread is less than the fees, do not place an order
    if spread < total_fees:
        logger.info(
            "Spread is less than total fees, Not a profitable opportunity to trade")
    else:
        # If we do not have a position, there are no open orders and spread is greater than the total fees, place a limit buy order at the buying price
        if current_position <= 0.01 and (not buy_order) and current_price > buying_price:
            buy_limit_order = await post_alpaca_order(buying_price, selling_price, 'buy')
            sell_order = False
            if buy_limit_order:  # check some attribute of buy_order to see if it was successful
                logger.info(
                    "Placed buy limit order at {0}".format(buying_price))

        # if we have a position, no open orders and the spread that can be captured is greater than fees, place a limit sell order at the sell_order_price
        if current_position >= 0.01 and (not sell_order) and current_price < sell_order_price:
            sell_limit_order = await post_alpaca_order(buying_price, selling_price, 'sell')
            buy_order = False
            if sell_limit_order:
                logger.info(
                    "Placed sell limit order at {0}".format(selling_price))

        # Cutting losses
        # If we have do not have a position, an open buy order and the current price is above the selling price, cancel the buy limit order
        logger.info("Threshold price to cancel any buy limit order: {0}".format(
                    sell_order_price * (1 + cut_loss_threshold)))
        if current_position <= 0.01 and buy_order and current_price > (sell_order_price * (1 + cut_loss_threshold)):
            trading_client.cancel_orders()
            buy_order = False
            logger.info(
                "Current price > Selling price. Closing Buy Limit Order, will place again in next check")
        # If we have do have a position and an open sell order and current price is below the buying price, cancel the sell limit order
        logger.info("Threshold price to cancel any sell limit order: {0}".format(
                    buy_order_price * (1 - cut_loss_threshold)))
        if current_position >= 0.01 and sell_order and current_price < (buy_order_price * (1 - cut_loss_threshold)):
            trading_client.cancel_orders()
            sell_order = False
            logger.info(
                "Current price < buying price. Closing Sell Limit Order, will place again in next check")


def get_positions():
    positions = trading_client.get_all_positions()

    return positions


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
