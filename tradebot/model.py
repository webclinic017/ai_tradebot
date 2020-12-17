import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
import pandas as pd
import os
from pytrends.request import TrendReq
from tradebot.news import Twitter_News
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from official.nlp import optimization

# TODO: Add support for tpu and raspberry pi

class Model:
    """ The base model class other models inherit from """
    def __init__(self):
        pass

class BERT(Model):
    """ Contains everything to load and train the sentiment analysis model """

    def __init__(self, batch_size=64, epochs=10, train_test_split=0.8):
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_test_split = train_test_split

    def load_data(self):
        """ Load the training data """

        def _preprocess(text, labels):
            """ One Hot encode the labels in the Dataset """
            # Create a tensorflow lookup table for the sentiment categories
            categories = tf.constant(['positive', 'neutral', 'negative'])
            indices = tf.range(len(categories), dtype=tf.int64)
            table_init = tf.lookup.KeyValueTensorInitializer(categories, indices)
            num_oov_buckets = 1
            table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)

            # one hot encode labels of dataset
            cat_indices = table.lookup(labels)
            labels_enc = tf.one_hot(cat_indices, depth=len(categories) + num_oov_buckets)

            return text, labels_enc
        
        # Load dataset from csv file
        dataset = tf.data.experimental.make_csv_dataset(
            './data/sentiment_data/train.csv', self.batch_size, column_names=['sentiment', 'text'],
            label_name='sentiment', header=True, shuffle=True
        )

        # Apply transformations specified in _preprocess function
        dataset = dataset.map(_preprocess)

        return dataset

    def load_model(self, fresh=False):
        """ Create an instance of a sentiment analysis model
        
        Parameters:
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
                net = tf.keras.layers.Dense(4, activation='sigmoid', name='output')(net)
                return tf.keras.Model(text_input, net)

            classifier_model = model()

            # Define loss, metrics and callbacks for model
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            metrics = tf.keras.metrics.CategoricalAccuracy()
            callbacks = []

            # TODO: Fix model training infinitely long
            steps_per_epoch = 600
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
                                    metrics=metrics)

            return classifier_model
        else:
            """ Load saved model """
            classifier_model = tf.keras.models.load_model('./models/fin_sentiment_bert')

            # Define loss, metrics and callbacks for model
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            metrics = tf.keras.metrics.CategoricalAccuracy()

            # TODO: Fix model training infinitely long
            steps_per_epoch = 600
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
                                    metrics=metrics)

            return classifier_model

    def train(self, train=None, test=None, val=None):
        """ Train the sentiment analysis model """

        # if no training data is passed load data from dataset
        if not train or not test or not val:
            train = self.load_data()
        
        classifier_model = self.load_model()

        # train model
        classifier_model.fit(x=train,
                            # batch_size=self.batch_size,
                            steps_per_epoch=500,
                            epochs=self.epochs,
                            use_multiprocessing=True)

        print(f"Model score: {classifier_model.evaluate(val)}")

        # save model
        classifier_model.save('./models/fin_sentiment_bert', include_optimizer=True)

    def predict(self, data):
        """ Use the sentiment analysis model to predict the sentiment for a new text """

        # Load model and return sentiment for input text
        classifier_model = self.load_model()
        return classifier_model.predict(x=data, batch_size=1)

class Prediction_Model(Model):
    def __init__(self):
        pass        