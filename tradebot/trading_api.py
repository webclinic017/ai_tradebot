from coinbase.wallet.client import Client
import os

class Coinbase():
    def __init__(self):
        self.client = Client(os.environ['COINBASE_API_KEY'], 
                             os.environ['COINBASE_API_SECRET'], 
                             api_version='2020-12-08')
        self.payment_methods = self.client.get_payment_methods()
        self.account = self.client.get_primary_account()
        self.payment_method = self.client.get_payment_methods()[0]

    def get_current_price(self, currency):
        return (self.client.get_buy_price(currency=currency), self.client.get_sell_price(currency=currency))

    def buy(self, amount, currency):
        return self.account.buy(amount=amount, currency=currency, payment_method=self.payment_method.id)

    def sell(self, sell, amount, currency):
        return self.account.sell(amount=amount, currency=currency, payment_method=self.payment_method.id)

    def get_price(self, date, currency):
        return self.client.get_spot_price(currency_pair = f'{currency}-EUR', date=date)

# API for buing shares