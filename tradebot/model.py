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

tf.get_logger().setLevel('ERROR')

BATCH_SIZE = 64
EPOCHS = 10
AUTOTUNE = tf.data.experimental.AUTOTUNE

KEYWORD_LIST = ["Blockchain", "Bitcoin", "Ethereum", "Cryptocurreny"]

"""
TODO:
- Build input pipeline
    - [x] read dataset from csv
    - [ ] preprocess labels

- build bert model
    - [ ] preprocess inputs
    - [ ] train bert & cassifier
"""

class BERT:
    def load_data(self, data=None):
        def _preprocess(text, labels):
            categories = tf.constant(['positive', 'neutral', 'negative'])
            indices = tf.range(len(categories), dtype=tf.int64)
            table_init = tf.lookup.KeyValueTensorInitializer(categories, indices)
            num_oov_buckets = 1
            table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)

            cat_indices = table.lookup(labels)
            labels_enc = tf.one_hot(cat_indices, depth=len(categories) + num_oov_buckets)
            # i = tf.constant(0, name="Index")
            # labels_enc = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
            # label = 0
            # for i in tf.range(len(labels)):
            #     if labels[i]=='negative':
            #         label = -1
            #     elif label[i]=='neutral':
            #         label = 0
            #     else: 
            #         label = 1

            #     label = tf.one_hot(
            #         label, 3, name='label', axis=-1)

            #     labels_enc.write(i, label)
            #     i = i+1

            #     print(f"-----MAP FUNC-----\nINDEX: {i}\nLABEL: {label}\nLABELS LIST: {labels_enc}")

            return text, labels_enc

        def model():
            text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
            preprocessing_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_cased_preprocess/2', name='preprocessing')
            encoder_inputs = preprocessing_layer(text_input)
            encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3', trainable=True, name='BERT_encoder')
            outputs = encoder(encoder_inputs)
            net = outputs['pooled_output']
            net = tf.keras.layers.Dropout(0.1)(net)
            net = tf.keras.layers.Dense(64, activation="relu")(net)
            net = tf.keras.layers.Dense(64, activation="relu")(net)
            net = tf.keras.layers.Dropout(0.1)(net)
            net = tf.keras.layers.Dense(4, activation='sigmoid', name='classifier')(net)
            return tf.keras.Model(text_input, net)

        raw_train_ds = tf.data.experimental.make_csv_dataset(
            './data/sentiment_data/train.csv', BATCH_SIZE, column_names=['sentiment', 'text'],
            label_name='sentiment', header=True
        )

        train_ds = raw_train_ds # .prefetch(buffer_size=AUTOTUNE)

        train_ds = train_ds.map(_preprocess)
        train_ds.shuffle(2500).batch(BATCH_SIZE)

        # for line in train_ds.take(1):
        #     print("-----AFTER MAP-----\n")
        #     print(f"THE LINE: {line}\n")
        #     print(f"THE TEXT LIST: {line[0]}\n")
        #     print(f"THE LABELS LIST: {line[1]}")

        classifier_model = model()

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics = tf.keras.metrics.CategoricalAccuracy()

        epochs = 5
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.1*num_train_steps)

        init_lr = 3e-5
        optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')

        classifier_model.compile(optimizer=optimizer,
                                 loss=loss,
                                 metrics=metrics)

        classifier_model.fit(x=train_ds.take(4000),
                             validation_data=train_ds.skip(4000),
                             epochs=EPOCHS)

        classifier_model.save('./models/fin_sentiment_bert', include_optimizer=False)

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

        df_tweets['sentiment'] = sentiment_scores