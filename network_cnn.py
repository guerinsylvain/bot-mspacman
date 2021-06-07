import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

class NetworkCNN:
    def __init__(self, obs_size, n_out):
        self._obs_size = obs_size
        self._n_out = n_out
        self._model = self.build_model()        

    @property
    def weights(self):
        return self._model.get_weights()

    @weights.setter
    def weights(self, weights):
        self._model.set_weights(weights)

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (8,8), strides=3, activation='relu', input_shape=self._obs_size),
            # tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            # tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            # tf.keras.layers.MaxPooling2D(2,2),
            # Flatten the results to feed into a DNN
            tf.keras.layers.Flatten(),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self._n_out, activation='relu')
        ])

        model.compile(Adam(lr=.001), loss=tf.keras.losses.Huber(), metrics=['accuracy'])
        print(model.summary())
        return model

    def compute(self, state, batch_size):
        return self._model.predict(state, batch_size=batch_size)

    def load_model(self, fileName):
        self._model = load_model(fileName)

    def save_model(self, fileName):
        self._model.save(fileName)

    def train(self, train_samples, train_labels, num_epochs, batch_size):
        return self._model.fit(train_samples, train_labels, epochs=num_epochs, verbose=0, batch_size=batch_size)
