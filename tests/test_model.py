import unittest
import tensorflow as tf
from tradebot import model

class Test_BERT(tf.test.TestCase):

    def setUp(self):
        super(Test_BERT, self).setUp()

    def tearDown(self):
        pass

    def test_data_loading(self):
        dataset = self.bert.load_data()