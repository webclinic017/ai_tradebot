import os
import datetime
import tweepy
import pandas as pd
from tqdm import tqdm

class Twitter_News():
    def __init__(self):
        self.auth = tweepy.AppAuthHandler(os.environ['TWITTER_API_KEY'], os.environ['TWITTER_API_SECRET'])
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True)

    def get_tweets(self):
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
        if not os.path.isfile('./data/sentiment_data/sentiment_tweets.csv'):
            df = pd.DataFrame(columns=['Date', 'Tweets'])

            print(f"FETCHING TWEETS")
            for user in tqdm(users):
                tweets=[]

                for tweet in tweepy.Cursor(self.api.user_timeline, screen_name=user, tweet_mode="extended", since=datetime.datetime(2015, 12, 13, 0, 0, 0)).items():
                    tweets.append([tweet.created_at.strftime("%Y-%m-%d"), tweet.full_text])

                df = df.append(pd.DataFrame(tweets, columns=['Date', 'Tweets']))
            
            df.to_csv('./data/sentiment_data/sentiment_tweets.csv', index=False)
            return df
        else:
            df = pd.read_csv('./data/sentiment_data/sentiment_tweets.csv')
            return df

class News_Headlines():
    def get_headlines(self):
        pass