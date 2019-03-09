from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image


class CNNImageClassifier:

    def __init__(self):
        # Init Sequentiel CNN Model
        self.classifier = Sequential()


    def convolutionAndReLU(self):
        # Define Convolution + Relu Step : 32 filters of 3x3, input image 64x64 RGB, apply ReLU method
        self.classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
        print('Init convolution ...')

    def pooling(self):
        # Max pool with 2x2 filters
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
        print('Init pooling ...')

    def flat(self):
        # Convert pooled image as 2-D array to one dimensional single vector
        self.classifier.add(Flatten())
        print('Init flatting ...')

    def fullyConnectedLayer(self):
        self.classifier.add(Dense(units=128, activation='relu'))
        self.classifier.add(Dense(units=1, activation='sigmoid'))
        print('Init Fully connected Layers ...')


    def compile(self):
        self.classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print('Compiling ...')

    def preprocess(self,training_path, test_path):
        print('Start preprocessing ...')
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        training_set = train_datagen.flow_from_directory(training_path,
                                                                   target_size=(64, 64),
                                                                   batch_size=32,
                                                                   class_mode='binary')
        test_set = test_datagen.flow_from_directory(test_path,
                                                              target_size=(64, 64),
                                                              batch_size=32,
                                                              class_mode='binary')
        self.classifier.fit_generator(training_set,
                                      steps_per_epoch=1589,
                                      epochs=25,
                                      validation_data=test_set,
                                      validation_steps=2000)
        print('Prepocessing images ...')

    def build(self,training_path, test_path):
        self.convolutionAndReLU()
        self.pooling()
        self.flat()
        self.fullyConnectedLayer()
        self.compile()
        self.preprocess(training_path, test_path)

    def prediction(self, image_path):
        print('Start prediction ...')
        test_image = image.load_img(image_path, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = self.classifier.predict(test_image)
        self.training_set.class_indices
        if result[0][0] == 1:
            prediction = 'dog'
        else:
            prediction = 'cat'
        print('IA predict that the image contain a ' + prediction)


if __name__ == '__main__':
    classifier = CNNImageClassifier()
    classifier.build('../data/imageclassifier/training_set', '../data/imageclassifier/test_set')
    classifier.prediction('../data/imageclassifier/test_set/cats/cat.4023.jpg')
