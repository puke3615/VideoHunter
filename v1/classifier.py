# coding=utf-8
import os

import scripts.love.data_handler
import utils
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
from keras.callbacks import ModelCheckpoint
from scripts.love import data_handler
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


class FaceClassifier:
    weights_path = utils.root_path('v1/weights/weights.h5')

    def __init__(self, weight_path=None, lr=1e-2, epoch=30):
        if weight_path is not None:
            self.weights_path = weight_path
        self.epoch = epoch
        self.model = self.build_model()
        self.compile_model(lr)

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(utils.IM_HEIGHT, utils.IM_WIDTH, 3)))
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
        model.add(Dense(7, activation='softmax'))
        return model

    def compile_model(self, lr):
        sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd)
        if os.path.exists(self.weights_path):
            self.model.load_weights(self.weights_path)
            print('Load weights.h5 successfully.')
        else:
            print('Model params not found.')

    def train(self, train_dir, classes=None, batch_size=32):
        file_num = utils.calculate_file_num(train_dir)
        steps_per_epoch = file_num // batch_size
        print('steps number is %d every epoch.' % steps_per_epoch)
        train_data_gen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        train_generator = train_data_gen.flow_from_directory(
            train_dir,
            classes=classes,
            target_size=(utils.IM_WIDTH, utils.IM_HEIGHT),
            batch_size=batch_size,
            class_mode='categorical')

        utils.ensure_dir(os.path.dirname(self.weights_path))
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            callbacks=[ModelCheckpoint(self.weights_path)],
            epochs=self.epoch
        )
        # self.model.fit(x, y, batch_size=32, epochs=self.epoch)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, batch_size=32)

    def predict(self, x, standard=True):
        if standard:
            x = x / 255.
        return self.model.predict(x)


PATH_TRAIN = utils.root_path('data/love/roles')
PATH_VAL = utils.root_path('data/love/roles')
# 是否训练
TRAIN = True
# 是否验证
VALIDATE = True

if __name__ == '__main__':
    print('Init model.')
    classifier = FaceClassifier(lr=1e-2, epoch=10)
    if TRAIN:
        print('Train model.')
        classifier.train(PATH_TRAIN, classes=utils.NAMES)

    if VALIDATE:
        names, index2name, name2index = utils.parse_name()
        x, y = data_handler.get_data(PATH_VAL, name2index, file_num=5000)
        prediction = classifier.predict(x, False)
        corrrect = np.equal(np.argmax(prediction, 1), np.argmax(y[:, :], 1))
        accuracy = np.mean(corrrect.astype(np.float32))
        print('Accuracy is %s' % accuracy)

        for name, prob in utils.parse_predict(prediction)[:10]:
            print(name, prob)
