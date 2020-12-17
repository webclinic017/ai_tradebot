import unittest
import tensorflow as tf
from tradebot import model

class Test_BERT(unittest.TestCase):
    def __init__(self):
        self.bert = model.BERT()

    def test_data_loading(self):
        dataset = self.bert.load_data()