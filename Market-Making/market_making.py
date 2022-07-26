# from alpaca_trade_api import Stream
from alpaca.data.live import CryptoDataStream
import config
import logging
import requests

# ENABLE LOGGING - options, DEBUG,INFO, WARNING?
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Alpaca API
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

HEADERS = {'APCA-API-KEY-ID': config.APCA_API_KEY_ID,
           'APCA-API-SECRET-KEY': config.APCA_API_SECRET_KEY}


client = CryptoDataStream(config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY)


async def quote_handler(data):
    # logger.info(data)
    # if data.exchange == 'FTXU':
    #     logger.info('--------Quote on FTXU--------')
    #     logger.info("Ask Price: {0}".format(data.ask_price))
    #     logger.info("Bid Price: {0}".format(data.bid_price))
    #     logger.info("Ask Size: {0}".format(data.ask_size))
    #     logger.info("Bid Size: {0}".format(data.bid_size))
    #     logger.info("Exchange: {0}".format(data.exchange))
    #     logger.info(
    #         "Bid-Ask Spread: {0}".format(data.ask_price - data.bid_price))
    if data.exchange == 'CBSE':
        logger.info('--------Quote on Coinbase---------')
        logger.info("Ask Price: {0}".format(data.ask_price))
        logger.info("Bid Price: {0}".format(data.bid_price))
        logger.info("Ask Size: {0}".format(data.ask_size))
        logger.info("Bid Size: {0}".format(data.bid_size))
        logger.info("Exchange: {0}".format(data.exchange))
        logger.info(
            "Bid-Ask Spread: {0}".format(data.ask_price - data.bid_price))


async def trade_handler(data):
    # logger.info(data)
    ftx_price = data.price if data.exchange == 'FTXU' else None
    cbse_price = data.price if data.exchange == 'CBSE' else None

    if data.exchange == 'FTXU':
        logger.info('--------Trade on FTXU--------')
        logger.info('Trade Price: {0}'.format(data.price))
        logger.info('Trade Size: {0}'.format(data.size))

    if data.exchange == 'CBSE':
        logger.info('--------Trade on Coinbase---------')
        logger.info('Trade Price: {0}'.format(data.price))
        logger.info('Trade Size: {0}'.format(data.size))

    if ftx_price and cbse_price:
        logger.info('--------Price difference on FTXU and Coinbase---------')
        logger.info('Price difference: {0}'.format(ftx_price - cbse_price))

trading_pair = 'BTCUSD'

client.subscribe_quotes(quote_handler, trading_pair)
# client.subscribe_trades(trade_handler, trading_pair)

client.run()

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
    '''

    pass
