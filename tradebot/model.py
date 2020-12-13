import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# TODO: Make available to all scripts
BATCH_SIZE = 12
EPOCHS = 10
BUFFER = 15000

class BERT:
    def __init__(self):
        pass

    def load_data(self):
        df = pd.read_csv('./data/finanical_news_sentiment.csv')
        labels = LabelEncoder().fit_transform(df['score'].to_numpy().reshape(-1, 1))

        # dataset = tf.data.experimental.make_csv_dataset('./data/finanical_news_sentiment.csv',
        #                                                 batch_size=BATCH_SIZE,
        #                                                 label_name='score',
        #                                                 header=True)

        # Load BERT and the preprocessing model from TF Hub.
        preprocess = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1')
        encoder = hub.load('https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1')

        sentiment = tf.keras.models.Sequential()
        sentiment.add(tf.keras.layers.Dense(128, input_shape=(128,), activation='relu'))
        sentiment.add(tf.keras.layers.Dense(64, activation='relu'))
        sentiment.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        sentiment.compile(loss='BinaryCrossentropy', 
                     optimizer=tf.keras.optimizers.Adam(1e-4), 
                     metrics=['acc'])

         # Use BERT on a batch of raw text inputs.
        # input = preprocess(df['text'].to_numpy())
        input = preprocess(df['text'].to_numpy())
        pooled_output = encoder(input)["pooled_output"]
        # print(pooled_output)

        sentiment.fit(pooled_output, labels, epochs=10, batch_size=BATCH_SIZE)

        # return dataset

    def train(self, dataset):
        self.model.train(
            dataset,
            epochs=10,
            batch_size=16,
            validation_split=0.1,
            erbose=1,
            shuffle=True
        )
        

# uses Sentiment, Trends to predict price
class Model:
    def __init__(self):
        self.model = tf.keras.models.Sequential()

        # TODO: Implement model
        self.model.add(tf.keras.models.Dense(32, input_shape=(3,), ))

        self.tensorboard = tf.keras.callbacks.tensorboard()
        self.early_stopping = tf.keras.callbacks.early_stopping()
        self.checkpoints = tf.keras.callbacks.checkpoints()

    def train(self, X_data, y_data, epochs, batch_size, validation_data):
        self.model.fit(X_data, y_data, 
                       epochs=epochs, 
                       batch_size=batch_size, 
                       validation_data=validation_data, 
                       callbacks=[self.tensorboard, self.early_stopping, self.checkpoints]
                       )

    def predict(self, X_data):
        self.model.predict(X_data)
        
    def save(self, path):
        self.model.save(path)