from crontab import CronTab
from tradebot.model import Prediction_Model

class MainThread():
    model = Prediction_Model()
    model.train()