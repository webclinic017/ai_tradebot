import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import os
from pytrends.request import TrendReq
from tradebot.news import Twitter_News
from tqdm import tqdm

BATCH_SIZE = 12
EPOCHS = 10
KEYWORD_LIST = ["Blockchain", "Bitcoin", "Ethereum", "Cryptocurreny"]

class BERT:
    def __init__(self):
        self.preprocess = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1')
        self.encoder = hub.load('https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1')

        self.sentiment = tf.keras.models.Sequential()
        self.sentiment.add(tf.keras.layers.Dense(64, input_shape=(128,), activation='relu'))
        self.sentiment.add(tf.keras.layers.Dense(64, activation='relu'))
        self.sentiment.add(tf.keras.layers.Dropout(0.2))
        self.sentiment.add(tf.keras.layers.Dense(32, input_shape=(128,), activation='relu'))
        self.sentiment.add(tf.keras.layers.Dense(32, activation='relu'))
        self.sentiment.add(tf.keras.layers.Dropout(0.2))
        # self.sentiment.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        self.sentiment.add(tf.keras.layers.Dense(3, activation='sigmoid'))

        self.sentiment.compile(loss='BinaryCrossentropy', 
                               optimizer=tf.keras.optimizers.Adam(1e-2), 
                               metrics=['acc'])

        if not os.listdir(path='./models'):
            self.train()
        else:
            self.load()

    def load_data(self):
        df = pd.read_csv('./data/finanical_news_sentiment.csv')
        labels = OneHotEncoder(sparse=False).fit_transform(df['score'].to_numpy().reshape(-1, 1))

        return df['text'].to_numpy(), labels

    # TODO: Add training for preprocessor, encoder
    def train(self):
        X, y = self.load_data()

        input = self.preprocess(X)
        pooled_output = self.encoder(input)["pooled_output"]

        self.sentiment.fit(pooled_output, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

        self.sentiment.save('./models/sentiment.h5')

    def load(self):
        self.sentiment = tf.keras.models.load_model('./models/sentiment.h5')

    def predict(self, data):
        input = self.preprocess(data)
        pooled_output = self.encoder(input)["pooled_output"]

        return self.sentiment.predict(pooled_output, batch_size=1)
        
class Model:
    def __init__(self):
        self.model = tf.keras.models.Sequential()

        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, input_shape=(3,), return_sequences=True, activation="relu")))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, activation="relu")))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, activation="relu")))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation="relu")))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(128, activation="relu"))
        self.model.add(tf.keras.layers.Dense(128, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(64, activation="relu"))
        self.model.add(tf.keras.layers.Dense(64, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    def load_data(self):
        pytrends = TrendReq(hl='en-US')
        pytrends.build_payload(KEYWORD_LIST, cat=0, timeframe='today 5-y', geo='', gprop='')
        trends = pytrends.interest_over_time()

        trends['searches'] = trends['Blockchain'] + trends['Bitcoin'] + trends['Ethereum'] + trends['Cryptocurreny']
        df = trends.drop(['Blockchain', 'Bitcoin', 'Ethereum', 'Cryptocurreny', 'isPartial'], axis=1)
        
        historical_prices_btc = pd.read_csv('./data/BTC-EUR.csv').drop(['Adj Close', 'Volume'], axis=1)
        historical_prices_eth = pd.read_csv('./data/ETH-EUR.csv').drop(['Adj Close', 'Volume'], axis=1)

        news = Twitter_News()
        df_tweets = news.get_tweets(users=['@decryptmedia', '@BTCTN', '@CryptoBoomNews', '@Cointelegraph', '@aantonop','@VentureCoinist', '@crypto', '@ForbesCrypto',])
        
        bert = BERT()
        sentiment_scores = []
        for tweet in tqdm(df_tweets['Tweets'].to_numpy()):
            sentiment_scores.append(bert.predict([tweet]))
            print("Sentiment Score " + str(bert.predict([tweet])))

        df_tweets['sentiment'] = sentiment_scores