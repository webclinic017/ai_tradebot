from tradebot import model

if __name__ == "__main__":
    sentiment_model = model.BERT()
    train = sentiment_model.load_data()
    # sentiment_model.train(train, validation=val)