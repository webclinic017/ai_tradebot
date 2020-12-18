from tradebot import model

if __name__ == "__main__":
    # model = model.BERT()
    # model.train()
    model = model.Prediction_Model()
    model.load_data()

    