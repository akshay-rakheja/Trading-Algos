from scripts.helpful_scripts import get_account
from brownie import interface, config, network
from scripts.get_weth import get_weth
from web3 import Web3

AMOUNT_TO_DEPOSIT = 0.1*10**18


def main():
    account = get_account()
    erc20_address = config["networks"][network.show_active()]["weth_token"]
    if network.show_active() in ["mainnet-fork"]:
        get_weth()
    # Get Lenging Pool from Aave
    lending_pool = get_lending_pool()
    # Approve ERC20 to Lending Pool
    approve_erc20(AMOUNT_TO_DEPOSIT, lending_pool.address,
                  erc20_address, account)
    # Deposit ERC20 to Lending Pool
    deposit_erc20(AMOUNT_TO_DEPOSIT, lending_pool, erc20_address, account)
    # Get User Account Data
    get_user_account_data(lending_pool, account)


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
    deposit_txn = lending_pool.deposit(
        erc20_address, amount, account.address, 0, {"from": account})
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
