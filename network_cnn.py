from tensorflow.keras.layers import AveragePooling2D, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU

class NetworkCNN:
    def __init__(self, obs_size, n_out):
        self._obs_size = obs_size
        self._n_out = n_out
        self._model = self.build_model()        

    def build_model(self):
        model = Sequential(name='frameset')
        model.add(Conv2D(32, kernel_size=(8, 8), input_shape=self._obs_size))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=(4, 4)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=(3, 3)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(AveragePooling2D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self._n_out))
        model.add(LeakyReLU(alpha=0.2))

        model.compile(Adam(lr=.0001), loss='mse', metrics=['accuracy'])

        print(model.summary())
        return model

    def compute(self, state, batch_size):
        return self.__model.predict(state, batch_size=batch_size)

    def load_model(self, fileName):
        self.__model = load_model(fileName)

    def save_model(self, fileName):
        self.__model.save(fileName)

    def train(self, train_samples, train_labels, num_epochs, batch_size):
        return self.__model.fit(train_samples, train_labels, epochs=num_epochs, verbose=0, batch_size=batch_size)
