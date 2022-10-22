from scripts.helpful_scripts import get_account
from brownie import interface, config, network
from brownie.network.gas.strategies import GasNowStrategy
from scripts.get_weth import get_weth
from web3 import Web3

AMOUNT_TO_DEPOSIT = 9*10**16
PERCENT_TO_BORROW = 0.9
gas_strategy = GasNowStrategy("fast")


def main():
    account = get_account()
    print("account: ", account.address)
    erc20_address = config["networks"][network.show_active()]["weth_token"]
    if network.show_active() in ["mainnet-fork"]:
        get_weth()
    # Get Lenging Pool address from Aave
    lending_pool = get_lending_pool()
    # Approve ERC20 to Lending Pool
    approve_erc20(AMOUNT_TO_DEPOSIT, lending_pool.address,
                  erc20_address, account)
    # Deposit ERC20 to Lending Pool
    deposit_erc20(AMOUNT_TO_DEPOSIT, lending_pool, erc20_address, account)
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
    usdt_address = config["networks"][network.show_active()]["usdt_token"]
    borrow_erc20(amount_to_borrow, lending_pool, usdt_address, account)
    # Get User Account Data
    buying_power, debt = get_user_account_data(lending_pool, account)
    # Get Alpaca ETH and USDT balance
    alpaca_address = config["networks"][network.show_active()]["alpaca_evm"]
    alpaca_balance(usdt_address, alpaca_address)
    # Deposit USDT to Alpaca
    alpaca_deposit_amount = amount_to_borrow
    alpaca_deposit(usdt_address, alpaca_address,
                   alpaca_deposit_amount, account)
    # Get Alpaca ETH and USDT balance after deposit
    alpaca_balance(usdt_address, alpaca_address)


def alpaca_balance(usdt_address, alpaca_address):
    web3 = Web3(Web3.HTTPProvider(
        config["networks"][network.show_active()]["provider"]))
    usdt = interface.IERC20(usdt_address)
    alpaca_usdt_balance = usdt.balanceOf(alpaca_address)
    alpaca_eth_balance = web3.eth.get_balance(alpaca_address)
    print(f"ETH balance on Alpaca: {alpaca_eth_balance/10**18}")
    print(f"USDT balance on Alpaca: {alpaca_usdt_balance/10**6}")


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
    print("Borrowed ERC20")
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
        f"Approving {amount} of ERC20: {erc20_address} to {lending_pool_address}")
    erc20 = interface.IERC20(erc20_address)
    approve_txn = erc20.approve(
        lending_pool_address, amount, {"from": account})
    approve_txn.wait(1)
    print("Approved ERC20")
    return approve_txn


def deposit_erc20(amount, lending_pool, erc20_address, account):
    print(
        f"Depositing {amount} of ERC20: {erc20_address} to {lending_pool.address}")
    # lending_pool = interface.ILendingPool(lending_pool.address)
    print("Erc20 address: ", erc20_address)
    print("amount: ", amount)
    print("account: ", account.address)
    deposit_txn = lending_pool.deposit(
        erc20_address, amount, account.address, 0, {"from": account, 'gas_limit': 1000000})
    deposit_txn.wait(1)
    print("Deposited ERC20")
    return deposit_txn


def get_lending_pool():
    lending_pool_addresses_provider = interface.ILendingPoolAddressesProvider(
        config["networks"][network.show_active()]["lending_pool_addresses_provider"])
    lending_pool_address = lending_pool_addresses_provider.getLendingPool()
    lending_pool = interface.ILendingPool(lending_pool_address)
    print(f"Lending pool address: {lending_pool.address}")
    return lending_pool
