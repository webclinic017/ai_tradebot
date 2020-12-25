import os
from coinbase.wallet.client import Client

class Coinbase():
    def __init__(self):
        self.client = Client(os.environ['COINBASE_API_KEY'], 
                             os.environ['COINBASE_API_SECRET'], 
                             api_version='2020-12-08')
        self.payment_methods = self.client.get_payment_methods()
        self.account = self.client.get_primary_account()
        self.payment_method = self.client.get_payment_methods()[0]

    def get_price(self, date, currency):
        return self.client.get_spot_price(currency_pair = f'{currency}-EUR', date=date)

    def get_portfolio(self):
        pass

    def buy(self, amount):
        pass

    def sell(self, amount):
        pass