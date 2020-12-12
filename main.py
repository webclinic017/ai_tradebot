from tradebot import training_data

if __name__ == "__main__":
    # coinbase = coinbase.Coinbase()
    # print(coinbase.get_current_price(currency='EUR'))
    model = training_data.Training_Data
    model.load_sentiment_data()