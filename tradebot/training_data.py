# https://towardsdatascience.com/google-trends-api-for-python-a84bc25db88f
from pytrends.request import TrendReq
import coinbase
from datetime import datetime
import pandas as pd

KEYWORDS = ['Blockchain', 'Cryptocurreny']

class Training_Data:

    def __init__(self):
        self.google_trends = TrendReq()
        self.google_trends.build_payload(KEYWORDS, cat=0, timeframe='today 5-y', geo='', gprop='')

        self.datetime = datetime.now()

    def load_sentiment_data():
        df = pd.read_csv('./data/financial_news_sentiment/all-data.csv', encoding='cp1252')
        return df

    def get_news(self):
        return articles

    def preprocess_data():

        
        # return (sentiment_data, trends, date), (price, sentiment_score)