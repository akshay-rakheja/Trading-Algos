from alpaca_trade_api import Stream
import config

client = Stream(config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY)


async def handler(data):
    print(data)

trading_pair = 'BTCUSD'

client.subscribe_crypto_orderbooks(handler, trading_pair)

client.run()
