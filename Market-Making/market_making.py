from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoQuotesRequest, CryptoTradesRequest
from alpaca.trading.requests import GetOrdersRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

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


trading_client = TradingClient(
    config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY, paper=True)

data_client = CryptoHistoricalDataClient()

trading_pair = 'ETH/USD'
notional_size = 20000
spread = 0.00
buying_price, selling_price = 0.00, 0.00
waitTime = 30


async def main():
    '''
    Get historical data from Alpaca and calculate RSI and Bollinger Bands.
    Backtest historical data to determine buy/sell/hold decisions and test performance.
    After backtesting, plot the results. Then, enter the loop to wait for new data and
    calculate entry and exit decisions.
    '''

    # closes all position AND also cancels all open orders
    trading_client.close_all_positions(cancel_orders=True)
    logger.info("Closed all positions")

    # # Log the current balance of the MATIC token in our Alpaca account
    # logger.info('BTC Position on Alpaca: {0}'.format(get_positions()))
    # # Log the current Cash Balance (USD) in our Alpaca account
    # global usd_position
    # usd_position = float(get_account_details()['cash'])
    # logger.info("USD position on Alpaca: {0}".format(usd_position))
    # # Get the historical data from Alpaca for backtesting
    # await get_crypto_bar_data(trading_pair, start_date, today, exchange)
    # # Add bar_data to a CSV for backtrader
    # bar_data.to_csv('bar_data.csv', index=False)
    # # Create and run a Backtest instance
    # await backtest_returns()

    while True:
        l1 = loop.create_task(get_crypto_bar_data(
            trading_pair))
        # Wait for the tasks to finish
        await asyncio.wait([l1])
        # Check if any trading condition is met
        await check_condition()
        # Wait for the a certain amount of time between each bar request
        await asyncio.sleep(waitTime)


async def get_crypto_bar_data(trading_pair):
    '''
    Get Crypto Bar Data from Alpaca for the last 10 minutes
    '''
    ten_mins_ago = datetime.now() - relativedelta(minutes=10)
    logger.info("Getting crypto bar data for {0} from {1}".format(
        trading_pair, ten_mins_ago))
    # Defining Bar data request parameters
    request_params = CryptoBarsRequest(
        symbol_or_symbols=[trading_pair],
        timeframe=TimeFrame.Minute,
        start=ten_mins_ago
    )
    # Get the bar data from Alpaca
    bars_df = data_client.get_crypto_bars(request_params).df
    # Calculate the order prices
    global buying_price, selling_price
    buying_price, selling_price = calc_order_prices(bars_df)

    return bars_df


def calc_order_prices(bars_df):
    max_high = bars_df['high'].max()
    min_low = bars_df['low'].min()
    mean_vwap = bars_df['vwap'].mean()

    buying_fee = 0.003*min_low
    selling_fee = 0.003*max_high

    global spread
    spread = max_high - min_low
    logger.info("Spread to capture: {0}".format(spread))
    logger.info("Mean VWAP: {0}".format(mean_vwap))
    logger.info("Min Low: {0}".format(min_low))
    logger.info("Max High: {0}".format(max_high))
    logger.info("Total Fees: {0}".format(buying_fee + selling_fee))

    selling_price = max_high*0.995
    buying_price = min_low*1.005
    return buying_price, selling_price


def get_positions():
    positions = trading_client.get_all_positions()
    return positions


# Post an Order to Alpaca
async def post_alpaca_order(price, side):
    '''
    Post an order to Alpaca
    '''
    try:
        if side == 'buy':
            limit_order_data = LimitOrderRequest(
                symbol="BTC/USD",
                limit_price=price,
                notional=notional_size,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.FOK
            )
            buy_limit_order = trading_client.submit_order(
                order_data=limit_order_data
            )
            logger.info(
                "Buy Limit Order placed for ETH/USD at : {0}".format(buy_limit_order.limit_price))
            return buy_limit_order
        else:
            limit_order_data = LimitOrderRequest(
                symbol="BTC/USD",
                limit_price=price,
                notional=notional_size,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.FOK
            )
            sell_limit_order = trading_client.submit_order(
                order_data=limit_order_data
            )
            logger.info(
                "Sell Limit Order placed for ETH/USD at : {0}".format(sell_limit_order.limit_price))
            return sell_limit_order

        # if order.status_code != 200:
        #     logger.info(
        #         "Undesirable response from Alpaca! {}".format(order.json()))
        #     return False
    except Exception as e:
        logger.exception(
            "There was an issue posting order to Alpaca: {0}".format(e))
        return False


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()


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
