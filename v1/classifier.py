import numpy as np
import love.data_handler as data_handler
import os
import tensorflow.examples.tutorials.mnist.input_data as input_data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD


def get_data():
    mnist = input_data.read_data_sets('/tmp/data/mnist', one_hot=True)
    x_train = mnist[0].images.reshape([-1, 28, 28, 1])
    y_train = mnist[0].labels
    x_test = mnist[2].images.reshape([-1, 28, 28, 1])
    y_test = mnist[2].labels
    return x_train, y_train, x_test, y_test


class Classifier:
    weights_path = 'weights/weights.h5'

    def __init__(self, weight_path=None, lr=1e-2, epoch=10):
        if weight_path is not None:
            self.weights_path = weight_path
        self.epoch = epoch
        self.model = self.build_model()
        self.compile_model(lr)

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        # model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        # model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8, activation='softmax'))
        return model

    def compile_model(self, lr):
        sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd)
        if os.path.exists(self.weights_path):
            self.model.load_weights(self.weights_path)
            print('Load weights.h5 successfully.')
        else:
            print('Model params not found.')


    def train(self, x, y):
        self.model.fit(x, y, batch_size=32, epochs=self.epoch)
        weights_dir = os.path.dirname(self.weights_path)
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        self.model.save_weights(self.weights_path)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, batch_size=32)

    def predict(self, x):
        return self.model.predict(x)


if __name__ == '__main__':
    print('Parse names.')
    names, index2name, name2index = data_handler.parse_name()
    print('Read train data.')
    x_train, y_train = data_handler.get_train_data('../love/roles', name2index, file_num=1000)

    print('Init model.')
    classifier = Classifier(lr=1e-2, epoch=10)
    # print('Train model.')
    # classifier.train(x_train, y_train)

    prediction = classifier.predict(x_train[:10, :, :, :])
    corrrect = np.equal(np.argmax(prediction, 1), np.argmax(y_train[:10, :], 1))
    accuracy = np.mean(corrrect.astype(np.float32))
    print('Accuracy is %s' % accuracy)

    for name, prob in data_handler.parse_predict(prediction):
        print(name, prob)
