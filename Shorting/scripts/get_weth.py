from scripts.helpful_scripts import get_account
from brownie import interface, config, network
from web3 import Web3


def main():
    # withdraw_eth()
    get_weth()
    # pass


def get_weth():
    account = get_account()
    weth = interface.IWeth(
        config["networks"][network.show_active()]["weth_token"])
    txn = weth.deposit({"from": account, "value": Web3.toWei(
        0.1, "ether"), "gas_price": Web3.toWei("1", "gwei")})
    txn.wait(1)
    print('Sent 0.1 ETH to WETH contract')
    return txn


def withdraw_eth():
    account = get_account()
    weth = interface.IWeth(
        config["networks"][network.show_active()]["weth_token"])
    txn = weth.withdraw(Web3.toWei(0.1, "ether"), {
                        "from": account,  "gas_price": Web3.toWei("1", "gwei")})
    txn.wait(1)
    print('Withdrawn 0.1 ETH from WETH contract')
    return txn
