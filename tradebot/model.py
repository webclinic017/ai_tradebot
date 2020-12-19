import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds
import data.sentiment_data.financial_sentiment_dataset
import numpy as np
import pandas as pd
import os
from pytrends.request import TrendReq
from tradebot.news import Twitter_News
from official.nlp import optimization
from tqdm import tqdm

# TODO: Add support for tpu

class Model:
    """ The base model class other models inherit from """
    pass

class BERT(Model):
    """ Contains everything to load and train the sentiment analysis model """

    def __init__(self, batch_size=64, epochs=10, train_test_split=0.8):
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_test_split = train_test_split

    def load_data(self):
        """ Load the training data """

        # builder = tfds.builder('financial_sentiment_dataset')
        # builder.download_and_prepare()

        train, test, val = tfds.load('financial_sentiment_dataset', split=['train[:80%]', 'train[80%:90%]', 'train[-10%:]'])

        for line in train.take(1):
            print(f"TRAIN DATASET: {line}")

        return train, test, val

    def load_model(self, fresh=False):
        """ Create an instance of a sentiment analysis model
        
        Arguments:
        - Fresh -> Returns a new model even if a pretrained model exists """

        # Check if a trained model exists
        if not os.path.exists('./models/fin_sentiment_bert') or fresh == True:
            """ No trainded model is saved """

            def model():
                """ Returns the sentiment analysis model """
                # The input for the preprocessing layer - normal test
                text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
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

            steps_per_epoch = 75
            num_train_steps = steps_per_epoch * self.epochs
            num_warmup_steps = int(0.1*num_train_steps)

            init_lr = 3e-5
            # AdamW optimizer
            optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                    num_train_steps=num_train_steps,
                                                    num_warmup_steps=num_warmup_steps,
                                                    optimizer_type='adamw')

            classifier_model.compile(optimizer=optimizer,
                                    loss=loss,
                                    metrics=metrics,
                                    experimental_steps_per_execution = 50)

            return classifier_model
        else:
            """ Load saved model """
            classifier_model = tf.keras.models.load_model('./models/fin_sentiment_bert')

            # Define loss, metrics and callbacks for model
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            metrics = tf.keras.metrics.CategoricalAccuracy()

            print(f"DATASET CARDINALITY: {tf.data.experimental.cardinality(train)}")

            steps_per_epoch = 75
            num_train_steps = steps_per_epoch * self.epochs
            num_warmup_steps = int(0.1*num_train_steps)

            init_lr = 3e-5
            # AdamW optimizer
            optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                      num_train_steps=num_train_steps,
                                                      num_warmup_steps=num_warmup_steps,
                                                      optimizer_type='adamw')

            classifier_model.compile(optimizer=optimizer,
                                    loss=loss,
                                    metrics=metrics,
                                    experimental_steps_per_execution = 50)

            return classifier_model

    def train(self, train=None, test=None, val=None):
        """ Train the sentiment analysis model """

        # if no training data is passed load data from dataset
        if train == None or test == None or val == None:
            (train, test, val) = self.load_data()
        
        classifier_model = self.load_model()

        # Define Callbacks
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir='logs', histogram_freq=1, write_graph=True,
            update_freq='epoch', profile_batch=2,
            embeddings_freq=0, embeddings_metadata=None,
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=1,
            restore_best_weights=True
        )

        # train model
        classifier_model.fit(x=train,
                            validation_data=val,
                            steps_per_epoch=75,
                            epochs=self.epochs,
                            use_multiprocessing=True,
                            callbacks=[tensorboard, early_stopping])

        print(f"Model score: {classifier_model.evaluate(test)}")

        # save model
        classifier_model.save('./models/fin_sentiment_bert', include_optimizer=True)

    def predict(self, data):
        """ Use the sentiment analysis model to predict the sentiment for a new text """

        # Load model and return sentiment for input text
        classifier_model = self.load_model()
        return classifier_model.predict(x=data, batch_size=1)

class Prediction_Model(Model):
    def __init__(self, batch_size=64, epochs=10, train_test_split=0.8):
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_test_split = train_test_split

    def load_data(self):
        if not os.path.isfile('./data/financial_data/sentiment_per_day.csv'):
            bert = BERT()
            tweets = Twitter_News().get_tweets()

            # tweets = news.News_Headlines().get_news()
            
            df = pd.DataFrame(columns=['date', 'tweet', 'sentiment'])
            
            print(f"PARSING TWEETS")
            for tweet in tqdm(tweets):
                df['date'] = tweet[0]
                df['tweet'] = tweet[1]
                sentiment = tf.argmax(bert.predict(tweet[1]), axis=0)
                if sentiment == 0:
                    df['sentiment'] = 'positive'
                elif sentiment == 1:
                    df['sentiment'] = 'neutral'
                else:
                    df['sentiment'] = 'negative'

            print(f"SENTIMENT PER TWEET: {df.head()}")

            df.to_csv('./data/financial_data/sentiment_per_tweet.csv')

            df_sentiment_day = pd.DataFrame(columns=['date', 'sentiment', 'total'])

            date = tweet[0]
            positive = 0
            neutral = 0
            negative = 0
            total = 0
            print("CREATE SENTIMENT PER DAY")
            for tweet in tqdm(df):
                sentiment = tweet[0]

                if date == tweet[0]:
                    if sentiment == 'negative':
                        negative = negative+1
                    elif sentiment == 'neutral':
                        neutral = neutral+1
                    else:
                        positive = positive+1
                    
                    total = total+1
                else:
                    df_sentiment_day['date'] = date
                    df_sentiment_day['sentiment'] = [
                        float(negative/total), float(neutral/total), float(positive/total)
                    ]
                    df_sentiment_day['total'] = total
                    date = tweet[0]

            print(f"SENTIMENT PER DAY: {df_sentiment_day.head()}")

            df_sentiment_day.to_csv('./data/financial_data/sentiment_per_day.csv')

        if not os.path.isfile('./data/financial_data/trends_per_day.csv'):
            pytrend = TrendReq()
            pytrend.build_payload(kw_list=['Cryptocurrency', 'Blockchain', 'Bitcoin', 'Ethereum'])

            region = pytrend.interest_by_region()
            trending = pytrend.trending_searches(pn='united_states')
            today = pytrend.today_searches(pn='US')
            related_queries = pytrend.related_queries()

            print(f"INTEREST BY REGION: \n{region.head()}\nTRENDING: \n{trending.head()}\nTODAY: \n{today.head()}\nRELATED QUERIES: \n{related_queries.head()}")

        dataset_historical_prices = tf.data.experimental.make_csv_dataset(
            [
                './data/financial_data/BTC-EUR.csv', 
                './data/financial_data/ETH-EUR.csv',
                './data/financial_data/sentiment_per_day.csv',
                './data/financial_data/trends_per_day.csv'
            ], 
            self.batch_size, header=True, shuffle=True
        )