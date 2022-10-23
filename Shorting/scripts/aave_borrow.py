from scripts.helpful_scripts import get_account
from brownie import interface, config, network
from brownie.network.gas.strategies import GasNowStrategy
from scripts.get_weth import get_weth
from web3 import Web3
import requests
import logging
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


# ENABLE LOGGING - options, DEBUG,INFO, WARNING?
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Alpaca Trading Client
trading_client = TradingClient(
    config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY, paper=True)

AMOUNT_TO_DEPOSIT = Web3.toWei(0.1, "ether")
PERCENT_TO_BORROW = 0.9
gas_strategy = GasNowStrategy("fast")


def main():
    account = get_account()
    print("account: ", account.address)
    weth_address = config["networks"][network.show_active()]["weth_token"]
    usdt_address = config["networks"][network.show_active()]["usdt_token"]
    alpaca_address = config["networks"][network.show_active()]["alpaca_evm"]
    get_balance(usdt_address, alpaca_address, account)
    if network.show_active() in ["mainnet-fork"]:
        print("Getting WETH")
        get_weth()
        get_balance(usdt_address, alpaca_address, account)
    # Get Lenging Pool address from Aave
    lending_pool = get_lending_pool()
    # Approve ERC20 to Lending Pool
    approve_erc20(AMOUNT_TO_DEPOSIT, lending_pool.address,
                  weth_address, account)
    # Deposit ERC20 to Lending Pool
    deposit_erc20(AMOUNT_TO_DEPOSIT, lending_pool, weth_address, account)
    # Get User Account Data
    buying_power, debt = get_user_account_data(lending_pool, account)
    # USDT price in ETH
    if network.show_active() in ["mainnet-fork"]:
        price_feed_address = config["networks"][network.show_active(
        )]["usdt_eth_price_feed"]
    elif network.show_active() in ["goerli"]:
        price_feed_address = config["networks"][network.show_active(
        )]["eth_usdt_price_feed"]
    print(
        f"Price feed address: {price_feed_address} on network {network.show_active()}")
    latest_price = get_usdt_price(price_feed_address)
    # Calculate amount of USDT to borrow
    amount_to_borrow = latest_price*buying_power*PERCENT_TO_BORROW*10**6
    # Borrow ERC20 from Lending Pool
    borrow_erc20(amount_to_borrow, lending_pool, usdt_address, account)
    # Get User Account Data
    buying_power, debt = get_user_account_data(lending_pool, account)
    # Get Alpaca ETH and USDT balance
    get_balance(usdt_address, alpaca_address, account)
    # Deposit USDT to Alpaca
    alpaca_deposit_amount = amount_to_borrow
    alpaca_deposit(usdt_address, alpaca_address,
                   alpaca_deposit_amount, account)
    # Get Alpaca ETH and USDT balance after deposit
    get_balance(usdt_address, alpaca_address, account)


def get_balance(usdt_address, alpaca_address, account):
    web3 = Web3(Web3.HTTPProvider(
        config["networks"][network.show_active()]["provider"]))
    weth_address = config["networks"][network.show_active()]["weth_token"]
    weth = interface.IERC20(weth_address)
    usdt = interface.IERC20(usdt_address)
    alpaca_usdt_balance = usdt.balanceOf(alpaca_address)
    alpaca_eth_balance = web3.eth.get_balance(alpaca_address)
    alpaca_weth_balance = weth.balanceOf(alpaca_address)
    user_usdt_balance = usdt.balanceOf(account.address)
    user_eth_balance = web3.eth.get_balance(account.address)
    user_weth_balance = weth.balanceOf(account.address)
    print(f"ETH balance on Alpaca: {alpaca_eth_balance/10**18}")
    print(f"USDT balance on Alpaca: {alpaca_usdt_balance/10**6}")
    print(f"WETH balance on Alpaca: {alpaca_weth_balance/10**18}")
    print(f"ETH balance on user: {user_eth_balance/10**18}")
    print(f"USDT balance on user: {user_usdt_balance/10**6}")
    print(f"WETH balance on user: {user_weth_balance/10**18}")


def alpaca_deposit(usdt_address, alpaca_address, alpaca_deposit_amount, account):
    usdt = interface.IERC20(usdt_address)
    txn = usdt.transfer(
        alpaca_address, alpaca_deposit_amount, {"from": account})
    txn.wait(1)
    return txn


def get_usdt_price(price_feed_address):
    if network.show_active() in ["mainnet-fork"]:
        usdt_eth_price_feed = interface.IAggregatorV3(
            price_feed_address)
        latest_price = 1/(usdt_eth_price_feed.latestRoundData()[1]/10**18)
    elif network.show_active() in ["goerli"]:
        eth_usdt_price_feed = interface.IAggregatorV3(
            price_feed_address)
        latest_price = eth_usdt_price_feed.latestRoundData()[1]/10**18
    print(f"Latest price: {latest_price}")
    return latest_price


def borrow_erc20(amount, lending_pool, usdt_address, account):
    print(
        f"Borrowing {amount} of ERC20: {usdt_address} from {lending_pool.address}")
    # lending_pool = interface.ILendingPool(lending_pool.address)
    usdt = interface.IERC20(usdt_address)
    borrow_txn = lending_pool.borrow(
        usdt_address, amount, 2, 0, account.address, {"from": account})
    borrow_txn.wait(1)
    print("Borrowed USDT")
    return borrow_txn


def get_user_account_data(lending_pool, account):
    print(f"Getting user account data for {account.address}")
    account_data = lending_pool.getUserAccountData(account.address)
    collateral = Web3.fromWei(account_data[0], "ether")
    debt = Web3.fromWei(account_data[1], "ether")
    borrowing_power = Web3.fromWei(account_data[2], "ether")
    health_factor = account_data[5]/10**18
    print(f"{collateral} ETH total collateral")
    print(f"{debt} ETH total debt")
    print(f"{borrowing_power} ETH borrowing power")
    print(f"{health_factor} health factor")
    return (float(borrowing_power), float(debt))


def approve_erc20(amount, lending_pool_address, erc20_address, account):
    print(
        f"Approving {amount} of WETH: {erc20_address} to {lending_pool_address}")
    erc20 = interface.IERC20(erc20_address)
    approve_txn = erc20.approve(
        lending_pool_address, amount, {"from": account})
    approve_txn.wait(1)
    print("Approved ERC20")
    return approve_txn


def deposit_erc20(amount, lending_pool, erc20_address, account):
    print(
        f"Depositing {amount} of WETH: {erc20_address} to {lending_pool.address}")
    # lending_pool = interface.ILendingPool(lending_pool.address)
    print("Erc20 address: ", erc20_address)
    print("amount: ", amount)
    print("account: ", account.address)
    print("lending_pool: ", lending_pool)
    deposit_txn = lending_pool.deposit(
        erc20_address, amount, account.address, 0, {"from": account, 'gas_limit': 1000000, "allow_revert": True})
    deposit_txn.wait(1)
    print("Deposited ERC20")
    return deposit_txn


def get_lending_pool():
    lending_pool_addresses_provider = interface.ILendingPoolAddressesProvider(
        config["networks"][network.show_active()]["lending_pool_addresses_provider"])
    lending_pool_address = lending_pool_addresses_provider.getLendingPool()
    lending_pool = interface.ILendingPool(lending_pool_address)
    print(
        f"Lending pool address: {lending_pool.address}")
    return lending_pool


# Post and Order to Alpaca
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
                qty=current_position,
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
