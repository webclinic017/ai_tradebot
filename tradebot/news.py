import tweepy
import os
import datetime
import pandas as pd

class Twitter_News:
    def __init__(self):
        self.auth = tweepy.AppAuthHandler(os.environ['TWITTER_API_KEY'], os.environ['TWITTER_API_SECRET'])
        self.api = tweepy.API(self.auth)

    def get_tweets(self, users):
        if not os.path.isfile('./data/sentiment_tweets.csv'):
            df = pd.DataFrame(columns=['Date', 'Tweets'])

            for user in users:

                tweets=[]

                for tweet in tweepy.Cursor(self.api.user_timeline, screen_name=user, tweet_mode="extended", since=datetime.datetime(2015, 12, 13, 0, 0, 0)).items():
                    tweets.append([tweet.created_at.strftime("%Y-%m-%d"), tweet.full_text])

                df = df.append(pd.DataFrame(tweets, columns=['Date', 'Tweets']))
            
            df.to_csv('./data/sentiment_tweets.csv', index=False)
            return df
        else:
            df = pd.read_csv('./data/sentiment_tweets.csv')
            return df
