from tradebot import model

if __name__ == "__main__":
    model = model.Model()
    train = model.load_data()
    # model.train(train, validation=val)