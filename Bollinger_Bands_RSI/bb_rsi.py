from dateutil.relativedelta import relativedelta
import config
import logging
import asyncio
import requests
import pandas as pd
from datetime import date, datetime
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from alpaca.data.historical import HistoricalDataClient
from alpaca.common.time import TimeFrame
import json
import backtrader as bt
import backtrader.feeds as btfeeds

# ENABLE LOGGING - options, DEBUG,INFO, WARNING?
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Alpaca API
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

HEADERS = {'APCA-API-KEY-ID': config.APCA_API_KEY_ID,
           'APCA-API-SECRET-KEY': config.APCA_API_SECRET_KEY}


# Alpaca client
client = HistoricalDataClient(
    config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY)

# Trading and Backtesting variables
trading_pair = 'BTCUSD'
exchange = 'FTXU'
one_year_ago = datetime.now() - relativedelta(years=1)
start_date = str(one_year_ago.date())
today = date.today()
today = today.strftime("%Y-%m-%d")
rsi_upper_bound = 70
rsi_lower_bound = 30
bollinger_window = 20

bar_data = 0
latest_bar_data = 0

# Wait time between each bar request -> 1 hour (3600 seconds)
waitTime = 3600
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
    # Get the historical data from Alpaca for backtesting
    await get_crypto_bar_data(trading_pair, start_date, today, exchange)
    # Add bar_data to a CSV for backtrader
    bar_data.to_csv('bar_data.csv', index=False)
    # Create and run a Backtest instance
    await backtest_returns()

    while True:
        l1 = loop.create_task(get_crypto_bar_data(
            trading_pair, start_date, today, exchange))
        # Wait for the tasks to finish
        await asyncio.wait([l1])
        # Check if any trading condition is met
        await check_condition()
        # Wait for the a certain amount of time between each bar request
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
            columns=["trade_count", "symbol", "timeframe", "exchange"], axis=1)

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


def get_daily_returns(df):
    df['daily_returns'] = df['close'].pct_change()
    return df


def get_cumulative_return(df):
    df['cumulative_return'] = df['daily_returns'].add(1).cumprod().sub(1)
    return df


def get_buy_hold_returns(df):
    df['buy_hold_returns'] = (df['cumulative_return'] + 1) * 10000
    return df


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


# Backtesting strategy
class BB_RSI_Strategy(bt.Strategy):
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        # print(self.datas[0].close[0])
        self.bband = bt.indicators.BBands(
            self.datas[0], period=20)
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=14)

        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)
            # self.bar_executed = len(self)
        self.order = None

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])
        if self.order:
            return
        if not self.position:
            if self.dataclose[0] < self.bband.lines.bot and self.rsi[0] < rsi_lower_bound:
                self.order = self.buy()
                self.log('BUY CREATED, %.2f' % self.dataclose[0])
        else:
            if self.dataclose[0] < self.bband.lines.bot and self.rsi[0] < rsi_lower_bound:
                self.order = self.sell()
                self.log('SELL CREATED, %.2f' % self.dataclose[0])


async def backtest_returns():

    cerebro = bt.Cerebro()
    data = btfeeds.GenericCSVData(
        dataname='bar_data.csv',

        fromdate=datetime(2021, 7, 9, 0, 0, 0, 0),
        todate=datetime(2022, 7, 8, 0, 0, 0, 0),

        nullvalue=0.0,

        dtformat=('%Y-%m-%dT%H:%M:%S%z'),
        timeframe=bt.TimeFrame.Minutes,
        compression=60,
        datetime=0,
        high=2,
        low=3,
        open=1,
        close=4,
        volume=5,
        openinterest=-1,
        rsi=7,
        bb_hi=11,
        bb_li=12

    )
    cerebro.broker.set_cash(100000.0)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=50)
    cerebro.adddata(data)
    cerebro.addstrategy(BB_RSI_Strategy)
    print("Starting Portfolio Value: ${}".format(cerebro.broker.getvalue()))

    cerebro.run()

    print("Final Portfolio Value: ${}".format(cerebro.broker.getvalue()))

    cerebro.plot()

    return


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
