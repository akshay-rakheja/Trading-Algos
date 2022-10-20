from scripts.helpful_scripts import get_account
from brownie import interface, config, network


def main():
    get_weth()
    # pass


def get_weth():
    account = get_account()
    weth = interface.IWeth(
        config["networks"][network.show_active()]["weth_token"])
    txn = weth.deposit({"from": account, "value": 0.1*10**18})
    txn.wait(1)
    print('Sent 0.1 ETH to WETH contract')
    return txn
