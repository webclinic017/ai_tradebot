import os
import datetime
import tweepy
import pandas as pd
import numpy as np
from tqdm import tqdm
from tradebot.trading_api import Coinbase
from pytrends.request import TrendReq
from GoogleNews import GoogleNews

class Twitter_News():
    def __init__(self):
        self.auth = tweepy.AppAuthHandler(os.environ['TWITTER_API_KEY'], os.environ['TWITTER_API_SECRET'])
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True)

    def get_tweets(self):
        if not os.path.isfile('./data/sentiment_data/tweets.csv'):
            users = [
                '@decryptmedia',
                '@BTCTN',
                '@CryptoBoomNews',
                '@Cointelegraph',
                '@aantonop',
                '@VentureCoinist',
                '@crypto',
                '@ForbesCrypto',
                '@FinancialNews',
                '@IBDinvestors',
                '@NDTVProfit',
                '@WSJ',
                '@FinancialXpress',
                '@WSJCentralBanks',
                '@FinancialTimes'
            ]
            
            df = pd.DataFrame(columns=['date', 'tweets'])

            print(f"FETCHING TWEETS")
            for user in tqdm(users):
                tweets=[]

                for tweet in tweepy.Cursor(self.api.user_timeline, screen_name=user, tweet_mode="extended", since=datetime.datetime(2015, 1, 1, 0, 0, 0)).items():
                    tweets.append([tweet.created_at.strftime("%Y-%m-%d %H:%M:%S"), tweet.full_text])

                df = df.append(pd.DataFrame(tweets, columns=['date', 'tweets']))
            
            df.to_csv('./data/sentiment_data/tweets.csv', index=False)
            return df
        else:
            df = pd.read_csv('./data/sentiment_data/tweets.csv')
            return df

class News_Headlines():
    def get_headlines(self):
        if not os.path.isfile('./data/sentiment_data/headlines.csv'):
            googlenews = GoogleNews(lang='en', start='01/01/2015',end='12/01/2020') # mm/dd/yyyy

            news = []

            keywords = [
                'Blockchain',
                'Cryptocurrency',
                'Bitcoin',
                'Etherium',
                'Stock Market',
                'Finance'
            ]
            for keyword in tqdm(keywords):
                googlenews.get_news(keyword)
                results = googlenews.results()

                for result in results:
                    news.append([result['datetime'], result['title']])

            df = pd.DataFrame(news, columns=['date', 'headline'])
            df.to_csv('./data/sentiment_data/headlines.csv', index=False)
            return df
        else:
            return pd.read_csv('./data/sentiment_data/headlines.csv')

class Crypto_Prices():
    def get_data(self):
        if not os.path.isfile('./data/financial_data/crypto_prices.csv'):
            dates = pd.date_range('2015-12-13', '2020-12-13', freq='H')
            coinbase = Coinbase()

            btc = []
            eth = []
            
            for i, date in tqdm(dates):
                price_btc = coinbase.get_price(date=str(date), currency='BTC')
                price_eth = coinbase.get_price(date=str(date), currency='ETH')
                btc.append([str(date), price_btc['amount']])
                eth.append([str(date), price_eth['amount']])

            df = pd.DataFrame(btc, columns=['date', 'btc'])
            eth_df = pd.DataFrame(eth, columns=['date', 'eth']).drop('date', axis=1)
            df['eth'] = eth_df['eth']
            df.to_csv('./data/financial_data/crypto_prices.csv', index=False)
            return df
        else:
            return pd.read_csv('./data/financial_data/crypto_prices.csv')
        
class Google_Trends():
    def get_data(self):
        if not os.path.isfile('./data/financial_data/google_trends.csv'):
            pytrends = TrendReq()

            searches = pytrends.get_historical_interest(
                keywords=['Cryptocurrency', 'Blockchain', 'Bitcoin', 'Ethereum'],
                year_start=2015,
                month_start=1,
                day_start=1,
                hour_start=0,
                year_end=2020,
                month_end=12,
                day_end=22,
                hour_end=0,
            )

            searches = pd.DataFrame(searches)
            searches.to_csv('./data/financial_data/google_trends.csv', index=False)
            return searches
        else:
            return pd.read_csv('./data/financial_data/google_trends.csv')