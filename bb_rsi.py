from dateutil.relativedelta import relativedelta
import config
import logging
import asyncio
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.utils import dropna
from alpaca.data.historical import HistoricalDataClient
from alpaca.common.time import TimeFrame
import json
import plotly.graph_objects as go
import plotly.express as px
import backtrader as bt
import backtrader.analyzer as btanalyzers


# ENABLE LOGGING - options, DEBUG,INFO, WARNING?
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Alpaca API
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'
ALPACA_DATA_URL = 'https://data.alpaca.markets'

HEADERS = {'APCA-API-KEY-ID': config.APCA_API_KEY_ID,
           'APCA-API-SECRET-KEY': config.APCA_API_SECRET_KEY}


# Alpaca client
client = HistoricalDataClient(
    config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY)

one_year_ago = datetime.now() - relativedelta(years=1)
# Trading variables
trading_pair = 'BTCUSD'
exchange = 'FTXU'
start_date = str(one_year_ago.date())
today = date.today()
today = today.strftime("%Y-%m-%d")
rsi_upper_bound = 60
rsi_lower_bound = 40
bollinger_window = 20

bar_data = 0
latest_bar_data = 0

# Wait time between each bar request -> 1 hour (3600 seconds)
waitTime = 60
active_position = False
btc_position = 0
usd_position = 0


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
    global usd_position
    usd_position = float(get_account_details()['cash'])
    logger.info("USD position on Alpaca: {0}".format(usd_position))
    # Get the historical data from Alpaca
    await get_crypto_bar_data(trading_pair, start_date, today, exchange)
    # Backtest the historical data

    # cerebro = bt.Cerebro()
    # cerebro.adddata(bar_data)
    # cerebro.addstrategy(BB_RSI_Strategy)
    # cerebro.broker.setcash(100000.0)
    # cerebro.addsizer(bt.sizers.PercentSizer, percents=20)
    # cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe_ratio')
    # cerebro.addanalyzer(btanalyzers.Transactions, _name='transactions')
    # cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
    # backtest = cerebro.run()

    # print("Broker value after backtesting is:\t", cerebro.broker.getvalue())
    # await backtest_returns(bar_data)

    # Plot Bollinger bands from start date to today
    # plot_signals()
    while True:
        l1 = loop.create_task(get_crypto_bar_data(
            trading_pair, start_date, today, exchange))
        # Wait for the tasks to finish
        await asyncio.wait([l1])
        # print(latest_bar_data)
        # print(bar_data)
        await check_condition()
        # Wait for the a certain amount of time between each quote request
        await asyncio.sleep(waitTime)


async def get_crypto_bar_data(trading_pair, start_date, end_date, exchange):
    '''
    Get bar data from Alpaca for a given trading pair and exchange
    '''
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
        bars = get_daily_returns(bars)
        bars = get_cumulative_return(bars)

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
    logger.info("Checking BTC position on Alpaca")
    global btc_position
    btc_position = float(get_positions())
    logger.info("Checking Buy/Sell conditions for Bollinger bands and RSI")
    logger.info("Latest Closing Price: {0}".format(
        latest_bar_data['close'].values[0]))
    logger.info("Latest Upper BB Value: {0}".format(
        latest_bar_data['bb_high'].values[0]))
    logger.info("Latest MAvg BB Value: {0}".format(
        latest_bar_data['bb_mavg'].values[0]))
    logger.info("Latest Lower BB Value: {0}".format(
        latest_bar_data['bb_low'].values[0]))
    logger.info("Latest RSI Value: {0}".format(
        latest_bar_data['rsi'].values[0]))

    if latest_bar_data.empty:
        logger.info("Unable to get latest bar data")
    # If bollinger high indicator is 1 and RSI is above the upperbound, then buy
    if ((latest_bar_data['bb_hi'].values[0] == 1) & (latest_bar_data['rsi'].values[0] > rsi_upper_bound) & (btc_position > 0)):
        # if True:
        logger.info(
            "Sell signal: Bollinger bands and RSI are above upper bound")
        sell_order = await post_alpaca_order(trading_pair, btc_position, 'sell', 'market', 'gtc')
        if sell_order['status'] == 'accepted':
            logger.info("Sell order successfully placed for {0} {1}".format(
                btc_position, trading_pair))
        elif (sell_order['status'] == 'pending_new'):
            logger.info("Sell order is pending.")
            logger.info("BTC Position on Alpaca: {0}".format(get_positions()))
        else:
            logger.info("Sell order status: {0}".format(sell_order))
    elif ((latest_bar_data['bb_li'].values[0] == 1) & (latest_bar_data['rsi'].values[0] < rsi_lower_bound) & (btc_position == 0)):
        logger.info("Buy signal: Bollinger bands and RSI are below lower bound")
        print(type(latest_bar_data['close'].values[0]))
        print(type(usd_position))
        qty_to_buy = (0.2 * usd_position) / latest_bar_data['close'].values[0]
        print("Qty to buy: ", qty_to_buy)
        buy_order = await post_alpaca_order(trading_pair, qty_to_buy, 'buy', 'market', 'gtc')
        if buy_order['status'] == 'accepted':
            logger.info("Buy order successfully placed for {0} {1}".format(
                qty_to_buy, trading_pair))
        elif (buy_order['status'] == 'pending_new'):
            logger.info("Buy order is pending.")
            logger.info("BTC Position on Alpaca: {0}".format(get_positions()))
        else:
            logger.info("Buy order status: {0}".format(buy_order))
    else:
        logger.info("Hold signal: Bollinger bands and RSI are within bounds")


# bars = get_crypto_bar_data('BTCUSD', '2022-01-01', today, 'FTXU')


def get_daily_returns(df):
    df['daily_returns'] = df['close'].pct_change()
    return df


# bars = get_daily_returns(bars)


def get_cumulative_return(df):
    df['cumulative_return'] = df['daily_returns'].add(1).cumprod().sub(1)
    return df


# bars = get_cumulative_return(bars)


def get_buy_hold_returns(df):
    df['buy_hold_returns'] = (df['cumulative_return'] + 1) * 10000
    return df


# bars = get_buy_hold_returns(bars)


# forward fill any missing data points in our buy & hold strategies
# and forward fill BTC_daily_return for missing data points
# bars[['buy_hold_returns', 'daily_returns', ]] = bars[[
#     'buy_hold_returns', 'daily_returns']].ffill()


# print(bars)


def get_bb(df):
    # calculate bollinger bands
    indicator_bb = BollingerBands(
        close=df["close"], window=bollinger_window, window_dev=2)
    # Add Bollinger Bands to the dataframe
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


def sell_points(df):
    return bar_data.loc[(bar_data['rsi'] > rsi_upper_bound) & (bar_data['bb_hi'] > 0) & (bar_data['rsi'].shift() < rsi_upper_bound) & (bar_data['bb_hi'].shift() == 0)]


def buy_points(df):
    return bar_data.loc[(bar_data['rsi'] < rsi_lower_bound) & (bar_data['bb_li'] > 0) & (bar_data['rsi'].shift() > rsi_lower_bound) & (bar_data['bb_li'].shift() == 0)]


# print("RSI IS >70 here:\n", bars.loc[bars['rsi'] > 70])
# print("BB bands are overbought here:\n", bars.loc[bars['bb_hi'] > 0])
# buying_points = buy_points(bars)
# selling_points = sell_points(bars)
# print("Buying Points are: \n", buying_points)
# print("Selling Points are: \n", selling_points)


# buying_points['order'] = 'buy'
# selling_points['order'] = 'sell'

# Combine buys and sells into 1 data frame
# orders = pd.concat([buying_points[['order']],
#                    selling_points[['order']]]).sort_index()

# new dataframe with market data and orders merged
# portfolio = pd.merge(bars, orders, how='outer',
#                      left_index=True, right_index=True)


async def backtest_returns(df):

    # Backtest of SMA crossover strategy
    active_position = False
    equity = 10000
    buying_points = buy_points(df)
    selling_points = sell_points(df)
    buying_points['order'] = 'buy'
    selling_points['order'] = 'sell'
    print(buying_points)
    print(selling_points)
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

        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'],
                      name='Buy', mode='markers', marker_color='green', marker_symbol=[49], marker_size=10))
        fig.add_trace(go.Scatter(x=selling_points['timestamp'], y=selling_points['close'],
                      name='Sell', mode='markers', marker_color='red', marker_symbol=[50], marker_size=10))

    fig = px.line(df[['trading_returns', 'buy_hold_returns']],
                  color_discrete_sequence=['green', 'blue'])
    fig.show()

    return df


# portfolio = backtest_returns(portfolio)

# print("Portfolio after backtesting: \n", portfolio)


def get_positions():
    '''
    Get positions on Alpaca
    '''
    try:
        positions = requests.get(
            '{0}/v2/positions'.format(ALPACA_BASE_URL), headers=HEADERS)
        logger.info('Alpaca positions reply status code: {0}'.format(
            positions.status_code))
        if positions.status_code != 200:
            logger.info(
                "Undesirable response from Alpaca! {}".format(positions.json()))
        if len(positions.json()) != 0:
            btc_position = positions.json()[0]['qty']
        else:
            btc_position = 0
        logger.info('BTC Position on Alpaca: {0}'.format(btc_position))
    except Exception as e:
        logger.exception(
            "There was an issue getting positions from Alpaca: {0}".format(e))
    return btc_position


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
async def post_alpaca_order(symbol, qty, side, type, time_in_force):
    '''
    Post an order to Alpaca
    '''
    try:
        order = requests.post(
            '{0}/v2/orders'.format(ALPACA_BASE_URL), headers=HEADERS, json={
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


class BB_RSI_Strategy(bt.Strategy):
    def __init__(self):
        self.position = False

    def next(self):
        if not self.position:
            if (bar_data['rsi'] < rsi_lower_bound) & (bar_data['bb_li'] > 0) & (bar_data['rsi'].shift() > rsi_lower_bound) & (bar_data['bb_li'].shift() == 0):
                self.buy()
        elif (bar_data['rsi'] > rsi_upper_bound) & (bar_data['bb_hi'] > 0) & (bar_data['rsi'].shift() < rsi_upper_bound) & (bar_data['bb_hi'].shift() == 0):
            self.close()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
