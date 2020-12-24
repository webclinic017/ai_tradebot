import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds
import data.sentiment_data.financial_sentiment_dataset
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

    def load_data(self):
        crypto_prices = Crypto_Prices().get_data()
        trends = Google_Trends().get_data()
        news = News_Headlines().get_headlines()
        tweets = Twitter_News().get_tweets()

        bert = BERT()
        tweets_sentiment = bert.predict(tweets.to_numpy()[:, 1])
        news_sentiment = bert.predict(news.to_numpy()[:, 1])

        tweets_sentiment = [tf.argmax(tweets_sentiment[i]).numpy() for i in tf.range(len(tweets_sentiment))]
        news_sentiment = [tf.argmax(news_sentiment[i]).numpy() for i in tf.range(len(news_sentiment))]

        tweets['sentiment'] = tweets_sentiment
        news['sentiment'] = news_sentiment

    def load_model(self):
        def model():
            net = tf.keras.layers.Input()
            net = tf.keras.layers.LSTM(250)(net)
            net = tf.keras.layers.LSTM(250)(net)
            net = tf.keras.layers.Dropout(0.2)(net)
            net = tf.keras.layers.LSTM(250)(net)
            net = tf.keras.layers.LSTM(250)(net)
            net = tf.keras.layers.Dropout(0.2)(net)

    def train(self):
        pass

    def predict(self):
        pass