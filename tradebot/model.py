import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds
import data.sentiment_data.financial_sentiment_dataset
import data.financial_data.bitcoin_prediction_dataset
import numpy as np
import pandas as pd
import os
from tradebot.news import Twitter_News, News_Headlines, Crypto_Prices, Google_Trends
from official.nlp import optimization
from tqdm import tqdm
from datetime import datetime

# TODO: Add support for tpu

class Model:
    """ The base model class other models inherit from """
    pass

class BERT(Model):
    """ Contains everything to load and train the sentiment analysis model """

    def __init__(self, batch_size=32, epochs=10, train_test_split=0.8):
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_test_split = train_test_split

    def load_data(self):
        """ Load the training data """

        train, test, val = tfds.load('financial_sentiment_dataset', split=['train[:80%]', 'train[80%:90%]', 'train[-10%:]'], batch_size=self.batch_size, as_supervised=True, shuffle_files=True)

        return train, test, val

    def load_model(self, train=None, fresh=False):
        """ Create an instance of a sentiment analysis model
        
        Arguments:
        - Fresh -> Returns a new model even if a pretrained model exists """

        # Check if a trained model exists
        if not os.path.exists('./models/fin_sentiment_bert') or fresh == True:
            """ No trainded model is saved """

            def model():
                """ Returns the sentiment analysis model """
                # The input for the preprocessing layer - normal test
                text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
                # Loading the preprocessing layer from keras
                preprocessing_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_cased_preprocess/2', name='preprocessing')
                # transforming the text to the shape the bert model expects
                encoder_inputs = preprocessing_layer(text_input)
                # bert model encoding the input text
                encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3', trainable=True, name='BERT_encoder')
                # Input for classifier model
                outputs = encoder(encoder_inputs)

                # Classifier model
                net = outputs['pooled_output']
                net = tf.keras.layers.Dropout(0.1)(net)
                net = tf.keras.layers.Dense(128, activation="relu", name='classifier_layer_1')(net)
                net = tf.keras.layers.Dense(128, activation="relu", name='classifier_layer_2')(net)
                net = tf.keras.layers.Dropout(0.1)(net)
                net = tf.keras.layers.Dense(64, activation="relu", name='classifier_layer_3')(net)
                net = tf.keras.layers.Dense(64, activation="relu", name='classifier_layer_4')(net)
                net = tf.keras.layers.Dropout(0.1)(net)
                net = tf.keras.layers.Dense(4, name='output')(net)
                return tf.keras.Model(text_input, net)

            classifier_model = model()

            # Define loss, metrics and callbacks for model
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            metrics = tf.keras.metrics.CategoricalAccuracy()

            steps_per_epoch = tf.data.experimental.cardinality(train) if fresh == None else 1
            num_train_steps = steps_per_epoch * self.epochs
            num_warmup_steps = int(0.1*int(num_train_steps))

            init_lr = 3e-5
            # AdamW optimizer
            optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                    num_train_steps=num_train_steps,
                                                    num_warmup_steps=num_warmup_steps,
                                                    optimizer_type='adamw')

            classifier_model.compile(optimizer=optimizer,
                                    loss=loss,
                                    metrics=metrics)

            return classifier_model
        else:
            """ Load saved model """
            classifier_model = tf.keras.models.load_model('./models/fin_sentiment_bert')

            # Define loss, metrics and callbacks for model
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            metrics = tf.keras.metrics.CategoricalAccuracy()

            steps_per_epoch = tf.data.experimental.cardinality(train) if fresh == None else 1
            num_train_steps = steps_per_epoch * self.epochs
            num_warmup_steps = int(0.1*int(num_train_steps))

            init_lr = 3e-5
            # AdamW optimizer
            optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                      num_train_steps=num_train_steps,
                                                      num_warmup_steps=num_warmup_steps,
                                                      optimizer_type='adamw')

            classifier_model.compile(optimizer=optimizer,
                                    loss=loss,
                                    metrics=metrics)

            return classifier_model

    def train(self, train=None, test=None, val=None):
        """ Train the sentiment analysis model """

        # if no training data is passed load data from dataset
        if train == None or test == None or val == None:
            (train, test, val) = self.load_data()
        
        classifier_model = self.load_model(train=train, fresh=True)

        # Define Callbacks
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir='./tensorboard/'+str(datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1, write_graph=True,
            update_freq='epoch', profile_batch='1,20',
            embeddings_freq=1,
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=0,
            restore_best_weights=False
        )

        # train model
        # TODO: Fix tensorboard error https://github.com/tensorflow/tensorflow/issues/43200
        classifier_model.fit(x=train,
                            validation_data=val,
                            epochs=self.epochs,
                            callbacks=[early_stopping])

        # save model
        classifier_model.save('./models/fin_sentiment_bert', include_optimizer=False, overwrite=True)
        
        print(f"Model score: {classifier_model.evaluate(test)}")

    def predict(self, data):
        """ Use the sentiment analysis model to predict the sentiment for a new text """

        # Load model and return sentiment for input text
        classifier_model = self.load_model()
        return classifier_model.predict(x=data, verbose=1)

class Prediction_Model(Model):
    def __init__(self, batch_size=64, epochs=10, train_test_split=0.8):
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_test_split = train_test_split

    def load_data(self, train=False, update=False):
        """ loads the data for the time series prediction model
        
        args:
        - train -> whether function should load training data or live data
        - update -> if True will update the csv the tensorflow dataset is constructed from """

        if train == False:
            crypto_prices = Crypto_Prices().get_realtime_data()
            trends = Google_Trends().get_realtime_data()
            news = News_Headlines().get_realtime_data()
            tweets = Twitter_News().get_realtime_data()

            # TODO: construct dataset with from tensor slices
            dataset = tf.data.Dataset.from_tensor_slices()

            return data

        if update == True:
            # Construct new csv for tfds with more data
            crypto_prices = Crypto_Prices().get_training_data()
            trends = Google_Trends().get_training_data()
            news = News_Headlines().get_training_data()
            tweets = Twitter_News().get_training_data()

            if not os.path.isfile('./data/sentiment_data/tweets_sentiment.csv'):
                bert = BERT()
                tweets_sentiment = bert.predict(tweets.to_numpy()[:, 1])
                news_sentiment = bert.predict(news.to_numpy()[:, 1])

                tweets['sentiment'] = [tf.argmax(tweets_sentiment[i]).numpy() for i in tf.range(len(tweets_sentiment))]
                news['sentiment'] = [tf.argmax(news_sentiment[i]).numpy() for i in tf.range(len(news_sentiment))]

                tweets.to_csv('./data/sentiment_data/tweets_sentiment.csv', index=False)
                news.to_csv('./data/sentiment_data/news_sentiment.csv', index=False)

            tweets = pd.read_csv('./data/sentiment_data/tweets_sentiment.csv')
            news = pd.read_csv('./data/sentiment_data/news_sentiment.csv')

            tweets['date'] = pd.to_datetime(tweets['date'])
            tweets = tweets.sort_values('date')
            news['date'] = pd.to_datetime(news['date'])
            news = news.sort_values('date')

            df = pd.merge(tweets, news, on='date', how='outer')

            sentiment_per_day = pd.DataFrame()

            prev_day = 0
            for date in df['date']:
                if date != prev_day:
                    if pd.isnull(date):
                        continue
                    df_day = df.loc[df['date'] == date]

                    # positive = 0, neutral = 1, negative = 2, other = 3
                    positive_per_day = df_day['sentiment_x'].value_counts().get(0, 0) + df_day['sentiment_y'].value_counts().get(0, 0)
                    neutral_per_day = df_day['sentiment_x'].value_counts().get(1, 0) + df_day['sentiment_y'].value_counts().get(1, 0)
                    negative_per_day = df_day['sentiment_x'].value_counts().get(2, 0) + df_day['sentiment_y'].value_counts().get(2, 0)
                    other_per_day = df_day['sentiment_x'].value_counts().get(3, 0) + df_day['sentiment_y'].value_counts().get(3, 0)

                    sentiment_per_day = sentiment_per_day.append({'date': date, 'positive': positive_per_day, 'neutral': neutral_per_day, 'negative': negative_per_day, 'other': other_per_day}, ignore_index=True)

                    prev_day = date

            df = pd.DataFrame(crypto_prices)
            df['trends_btc'] = trends['Bitcoin'].to_numpy()[:43849]
            df['trends_eth'] = trends['Ethereum'].to_numpy()[:43849]
            df['trends_total'] = [i[0] + i[1] + i[2] + i[3] + i[4] for i in trends.to_numpy()[:43849]]

            df['date'] = pd.to_datetime(df['date'])
            df = pd.merge(df, sentiment_per_day, how='outer', on='date')
            df.fillna(0, inplace=True)

            print(df.head())

            df.to_csv('./data/financial_data/price_prediction_dataset.csv', index=False)
            
        train, test, val = tfds.load('bitcoin_prediction_dataset', split=['train[:80%]', 'train[80%:90%]', 'train[-10%:]'], batch_size=self.batch_size, shuffle_files=True)

        return train, test, val

    def load_model(self, fresh=False):

        if not os.path.exists('./models/fin_timeseries_model') or fresh == True:
            # TODO: Research on best time series network and implement
            def model():
                net = tf.keras.layers.Input(shape=(2, 5))
                net = tf.keras.layers.LSTM(250)(net)
                net = tf.keras.layers.LSTM(250)(net)
                net = tf.keras.layers.Dropout(0.2)(net)
                net = tf.keras.layers.LSTM(250)(net)
                net = tf.keras.layers.LSTM(250)(net)
                net = tf.keras.layers.Dropout(0.2)(net)
                return tf.keras.Model(input, net)

            model = model()

            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr=1e-4), 
                loss=tf.keras.losses.MSE(), 
                metrics=['acc'])

            return model
        else:
            model = tf.kers.models.load_model('./models/fin_timeseries_model')

            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr=1e-4), 
                loss=tf.keras.losses.MSE(), 
                metrics=['acc'])

            return model

    def train(self, train=None, test=None, val=None):
        if train == None or test == None or val == None:
            (train, test, val) = self.load_data(train=True)

        model = self.load_model()

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f"./tensorboard/pred/{datetime.now()}")
        early_stopping = tf.keras.callbacks.EarlyStopping()

        model.fit(
            x=train,
            validation_data=val,
            epochs=epochs,
            callbacks=[tensorboard, early_stopping]
        )

        model.save('./models/fin_timeseries_model', include_optimizer=False, overwrite=True)

        print(f"Model score: {model.evaluate(test)}")

    def predict(self, data):
        model = self.load_model()
        model.predict(x=data, verbose=1)