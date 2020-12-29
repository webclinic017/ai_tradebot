import tradebot
from tradebot import model

if __name__ == "__main__":
    # tradebot()
    model = model.Prediction_Model()
    model.load_data(train=True)