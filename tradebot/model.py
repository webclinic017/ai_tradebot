import tensorflow as tf

class Sentiment_Model:
    def __init__(self, loss, optimizer):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Embedding(<size>))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dense(32, activation='relu'))
        self.model.add(tf.keras.layers.Dense(3, activation='softmax'))

        self.model.compile(loss='categotical_cross_entropy', 
                           optimizer=tf.keras.optimizer.Adam(1e-4), 
                           metrics=['accuracy'])

        self.tensorboard = tf.keras.callbacks.tensorboard()
        self.early_stopping = tf.keras.callbacks.early_stopping()
        self.checkpoints = tf.keras.callbacks.checkpoints()

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def train(self, X_data, y_data, epochs, batch_size, validation_data):
        self.model.fit(X_data, y_data, 
                       epochs=epochs, 
                       batch_size=batch_size, 
                       validation_data=validation_data, 
                       callbacks=[self.tensorboard, self.early_stopping, self.checkpoints])

    def predict(self, X_data):
        self.model.predict(X_data)
        
    def save(self, path):
        self.model.save(path)

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
                       callbacks=[self.tensorboard, self.early_stopping, self.checkpoints])

    def predict(self, X_data):
        self.model.predict(X_data)
        
    def save(self, path):
        self.model.save(path)