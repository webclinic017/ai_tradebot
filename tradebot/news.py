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
    """ A class for the twitter api """

    def __init__(self):
        # Authenticate to the twitter api
        self.auth = tweepy.AppAuthHandler(os.environ['TWITTER_API_KEY'], os.environ['TWITTER_API_SECRET'])
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True)

    def get_training_data(self):
        """ return training data from twitter api """

        # check if data has been downloaded
        if not os.path.isfile('./data/sentiment_data/tweets.csv'):
            # users from which data is being downloaded
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

            # fetching tweets for evers user in list
            for user in tqdm(users):
                tweets=[]

                # downloading every tweet from user timeline and appending it to list tweets
                for tweet in tweepy.Cursor(self.api.user_timeline, screen_name=user, tweet_mode="extended", since=datetime.datetime(2015, 1, 1, 0, 0, 0)).items():
                    tweets.append([tweet.created_at.strftime("%Y-%m-%d %H:%M:%S"), tweet.full_text])

                # appending the user timeline tweets to dataframe
                df = df.append(pd.DataFrame(tweets, columns=['date', 'tweets']))
            
            # saving dataframe to csv
            df.to_csv('./data/sentiment_data/tweets.csv', index=False)
            return df
        else:
            df = pd.read_csv('./data/sentiment_data/tweets.csv')
            return df

    def get_realtime_data(self):
        pass

class News_Headlines():
    """ A class for news headlines using google news """

    def get_training_data(self):
        """ load training data from google news """

        # check if data has been downloaded
        if not os.path.isfile('./data/sentiment_data/headlines.csv'):
            googlenews = GoogleNews(lang='en', start='01/01/2015') # mm/dd/yyyy

            news = []

            keywords = [
                'Blockchain',
                'Cryptocurrency',
                'Bitcoin',
                'Etherium',
                'Stock Market',
                'Finance'
            ]

            # fetch news headlines for every keyword in keywords list
            for keyword in tqdm(keywords):
                googlenews.get_news(keyword)
                results = googlenews.results()

                # append news headlines to list news
                for result in results:
                    news.append([result['datetime'], result['title']])

            # create a pandas dataframe with news list and save it to csv
            df = pd.DataFrame(news, columns=['date', 'headline'])
            df.to_csv('./data/sentiment_data/headlines.csv', index=False)
            return df
        else:
            return pd.read_csv('./data/sentiment_data/headlines.csv')

    def get_realtime_data(self):
        pass

class Crypto_Prices():
    """ A class for loading crypto currency prices """

    def get_training_data(self):
        """ load training data from coinbase """

        # check if data has been downloaded
        if not os.path.isfile('./data/financial_data/crypto_prices.csv'):
            # create a pandas dataframe with dates from last five years
            dates = pd.date_range('2015-12-13', datetime.now().strftime('%Y-%m-%d %H:%M%S'), freq='H')
            coinbase = Coinbase()

            prices = []
            
            # fetch prices for every date in dataframe and append to prices array
            for date in tqdm(dates):
                price_btc = coinbase.get_price(date=str(date), currency='BTC')
                price_eth = coinbase.get_price(date=str(date), currency='ETH')
                prices.append([str(date), price_btc['amount'], price_eth['amount']])

            # create Dataframe from prices list and save to csv
            df = pd.DataFrame(prices, columns=['date', 'btc', 'eth'])
            df.to_csv('./data/financial_data/crypto_prices.csv', index=False)
            return df
        else:
            return pd.read_csv('./data/financial_data/crypto_prices.csv')

    def get_realtime_data(self):
        pass
        
class Google_Trends():
    """ A class for loading google trends data """

    def get_training_data(self):
        """ load training data from google trends """

        # check if data has been downloaded
        if not os.path.isfile('./data/financial_data/google_trends.csv'):
            pytrends = TrendReq()

            # load searches per hour from google trends for the last five years
            searches = pytrends.get_historical_interest(
                keywords=['Cryptocurrency', 'Blockchain', 'Bitcoin', 'Ethereum'],
                year_start=2015,
                month_start=1,
                day_start=1,
                hour_start=0,
                year_end=datetime.now().strftime('%Y'),
                month_end=datetime.now().strftime('%m'),
                day_end=datetime.now().strftime('%d'),
                hour_end=datetime.now().strftime('%H'),
            )

            # make dataframe with trends and save to csv
            searches = pd.DataFrame(searches)
            searches.to_csv('./data/financial_data/google_trends.csv', index=False)
            return searches
        else:
            return pd.read_csv('./data/financial_data/google_trends.csv')

    def get_realtime_data():
        pass