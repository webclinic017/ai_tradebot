from coinbase.wallet.client import Client
import os

class Coinbase:

    # TODO: Adapt to currency
    MAX_BUY_AMOUNT=0.00001
    MAX_SELL_AMOUNT=0.00001

    # TODO: Add daily / monthly limits

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
        if amount < MAX_BUY_AMOUNT:
            return self.account.buy(amount=amount, currency=currency, payment_method=self.payment_method.id)
        else:
            raise Exeption("The amount you where trying to buy exeeded your buying amount threshold.")

    def sell(sell, amount, currency):
        if amount < MAX_SELL_AMOUNT:
            return self.account.sell(amount=amount, currency=currency, payment_method=self.payment_method.id)
        else:
            raise Exeption("The amount you where trying to sell exeeded your selling amount threshold.")
