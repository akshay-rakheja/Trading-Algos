from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoQuotesRequest, CryptoTradesRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from alpaca.trading.client import TradingClient
import logging
import config
import asyncio

# ENABLE LOGGING - options, DEBUG,INFO, WARNING?
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Alpaca API
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

HEADERS = {'APCA-API-KEY-ID': config.APCA_API_KEY_ID,
           'APCA-API-SECRET-KEY': config.APCA_API_SECRET_KEY}


# stream_client = CryptoDataStream(
#     config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY)
trading_client = TradingClient(
    config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY, paper=True)

client = CryptoHistoricalDataClient()


one_hour_ago = datetime.now() - relativedelta(minute=10)
print(one_hour_ago)

request_params = CryptoBarsRequest(
    symbol_or_symbols=["ETH/USD"],
    timeframe=TimeFrame.Minute,
    start=one_hour_ago
)


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


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()

bars_df = client.get_crypto_bars(request_params).df


max_high = bars_df['high'].max()
min_low = bars_df['low'].min()
mean_vwap = bars_df['vwap'].mean()
mean_close = bars_df['close'].mean()

buying_fee = 0.003*min_low
selling_fee = 0.003*max_high

print("Total fees: {0}".format(buying_fee + selling_fee))

diff = max_high - min_low
print("Spread: {0}".format(diff))

print("Mean Close: {0}".format(mean_close))
print("Mean VWAP: {0}".format(mean_vwap))
print("Max High: ", max_high)
print("Min Low: ", min_low)


async def quote_handler(data):
    # logger.info(data)
    if data.exchange == 'FTXU':
        logger.info('--------Quote on FTXU--------')
        logger.info("Ask Price: {0}".format(data.ask_price))
        logger.info("Bid Price: {0}".format(data.bid_price))
        logger.info("Ask Size: {0}".format(data.ask_size))
        logger.info("Bid Size: {0}".format(data.bid_size))
        logger.info("Exchange: {0}".format(data.exchange))
        logger.info(
            "Bid-Ask Spread: {0}".format(data.ask_price - data.bid_price))
    # if data.exchange == 'CBSE':
    #     logger.info('--------Quote on Coinbase---------')
    #     logger.info("Ask Price: {0}".format(data.ask_price))
    #     logger.info("Bid Price: {0}".format(data.bid_price))
    #     logger.info("Ask Size: {0}".format(data.ask_size))
    #     logger.info("Bid Size: {0}".format(data.bid_size))
    #     logger.info("Exchange: {0}".format(data.exchange))
    #     logger.info(
    #         "Bid-Ask Spread: {0}".format(data.ask_price - data.bid_price))


async def trade_handler(data):
    # logger.info(data)
    ftx_price = data.price if data.exchange == 'FTXU' else None
    cbse_price = data.price if data.exchange == 'CBSE' else None

    # if data.exchange == 'FTXU':
    #     logger.info('--------Trade on FTXU--------')
    #     logger.info('Trade Price: {0}'.format(ftx_price))
    #     logger.info('Trade Size: {0}'.format(data.size))

    if data.exchange == 'CBSE':
        logger.info('--------Trade on Coinbase---------')
        logger.info('Trade Price: {0}'.format(cbse_price))
        logger.info('Trade Size: {0}'.format(data.size))

    if ftx_price and cbse_price:
        logger.info('--------Price difference on FTXU and Coinbase---------')
        logger.info('Price difference: {0}'.format(ftx_price - cbse_price))

trading_pair = 'BTC/USD'


quote_request = CryptoQuotesRequest(trading_pair)
print(quote_request)


async def submit_order(symbol, qty, side, type, time_in_force):
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
                'client_order_id': 'market_making_strategy'
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


def market_making_conditions():
    '''
    Check the market conditions to see what limit orders to place

    -- if the cbse trade price is greater than ftx bid price, place a buy limit order at the cbse bid price on ftx

    Alternate strategy:
    - calculate moving average of closes, highs and lows of cbse for last 30 minutes
    - place buy limit order at the average of the lows and sell limit orders at the average of the highs
    - if ftx trade price is outside the average of highs and lows of cbse, close all positions until the price is within the average range

    '''

    pass
