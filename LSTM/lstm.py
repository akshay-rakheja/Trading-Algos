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
qty_to_trade = 5
# Wait time between each bar request
waitTime = 600
data = 0

current_position, current_price = 0, 0
predicted_price = 0


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
        global predicted_price
        predicted_price = pred.predictModel()
        logger.info("Predicted Price is {0}".format(predicted_price))
        l1 = loop.create_task(check_condition())
        await asyncio.wait([l1])
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
                 trading_pair: str = 'ETHUSD',
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
        # Alpaca Trading Client
        trading_client = TradingClient(
            config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY, paper=True)

        # Alpaca Market Data Client
        data_client = CryptoHistoricalDataClient()

        time_diff = datetime.now() - relativedelta(hours=1000)
        logger.info("Getting crypto bar data for {0} from {1}".format(
            self.trading_pair, time_diff))
        # Defining Bar data request parameters
        request_params = CryptoBarsRequest(
            symbol_or_symbols=[trading_pair],
            timeframe=TimeFrame.Hour,
            start=time_diff
        )
        # Get the bar data from Alpaca
        df = data_client.get_crypto_bars(request_params).df
        global current_price
        current_price = df.iloc[-1]['close']
        return df

    def getFeature(self):
        df = self.getAllData()
        data = df.filter([self.feature])
        data = data.values
        return data

    def scaleData(self):
        data = self.getFeature()
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler

    # train on all data for which labels are available (train + test from dev)
    def getTrainData(self):
        scaled_data = self.scaleData()[0]
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
        logger.info("-----Training model-----")
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

        logger.info("-----Predicting Price-----")
        pred_true = scaler.inverse_transform(pred)
        return pred_true[0][0]


async def check_condition():
    '''
    Strategy:
    - If the predicted price an hour from now is above the current price and we do not have a position, buy
    - If the predicted price an hour from now is below the current price and we do have a position, sell
    '''
    global current_position, current_price, predicted_price
    current_position = get_positions()
    logger.info("Current Price is: {0}".format(current_price))
    logger.info("Current Position is: {0}".format(current_position))
    # If we do not have a position and current price is less than the predicted price place a market buy order
    if float(current_position) <= 0.01 and current_price < predicted_price:
        logger.info("Placing Buy Order")
        buy_order = await post_alpaca_order('buy')
        if buy_order:  # check some attribute of buy_order to see if it was successful
            logger.info("Buy Order Placed")

    # If we do have a position and current price is greater than the predicted price place a market sell order
    if current_position >= 0.01 and current_price > predicted_price:
        logger.info("Placing Sell Order")
        sell_order = await post_alpaca_order('sell')
        if sell_order:
            logger.info("Sell Order Placed")


async def post_alpaca_order(side):
    '''
    Post an order to Alpaca
    '''
    try:
        if side == 'buy':
            market_order_data = MarketOrderRequest(
                symbol="ETHUSD",
                qty=qty_to_trade,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )
            buy_order = trading_client.submit_order(
                order_data=market_order_data
            )
            return buy_order
        else:
            market_order_data = MarketOrderRequest(
                symbol="ETHUSD",
                qty=qty_to_trade,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            sell_order = trading_client.submit_order(
                order_data=market_order_data
            )
            return sell_order
    except Exception as e:
        logger.exception(
            "There was an issue posting order to Alpaca: {0}".format(e))
        return False


def get_positions():
    positions = trading_client.get_all_positions()
    # print(positions[0])
    global current_position
    for p in positions:
        if p.symbol == 'ETHUSD':
            current_position = p.qty
            return current_position
    return current_position


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
