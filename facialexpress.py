from fer import FerData
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.preprocessing import image
from keras import backend as bk
import keras as K
import numpy as np
import h5py
import os

class FacialExpress:
    ''' Facial expression detection
    '''
    
    def __init__(self):
        '''
        '''
        self._models = {
            "mnist_cnn"   : (self.mnist_cnn, 128, 16),
            "simple_cnn" : (self.simple_cnn, 128, 16),
            'vgg16'     : (self.vgg16, 128, 12),
        }

        self._model = None
        self._data_folder = "data"
        if not os.path.exists(self._data_folder):
            os.makedirs(self._data_folder)

    def load_saved_model(self, model_name='simple_cnn'):
        '''
        '''
        with open(os.path.join(self._data_folder, model_name + '.json'), 'r') as jf:
            jm = jf.read()

        model = model_from_json(jm)
        model.load_weights(os.path.join(self._data_folder, model_name + ".h5"))

        model.compile(loss=K.losses.categorical_crossentropy,
                      optimizer=K.optimizers.Adadelta(),
                      metrics=['accuracy'])

        self._model = model
        return self._model

    def predict(self, img_file, model=None):
        if None == model:
            model = self._model
        if None == model:
            model = self.load_saved_model()

        img = image.load_img(img_file, color_mode='grayscale', target_size=(FerData.IMG_WIDTH, FerData.IMG_HEIGHT))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        properbilities = model.predict(x)
        predictions = [p for p in properbilities]
        print(predictions)

    def make_model(self, model_name="simple_cnn"):
        '''
        '''
        bk.set_image_data_format('channels_last')
        (train_label, train_img), (test_label, test_img) = FerData().load_data()

        self._train_img = train_img.astype('float32') / 255
        self._test_img = test_img.astype('float32') / 255

        self._classes_num = 7
        self._train_label = K.utils.to_categorical(train_label, self._classes_num)
        self._test_label = K.utils.to_categorical(test_label, self._classes_num)

        self._input_shape = (FerData.IMG_HEIGHT, FerData.IMG_WIDTH, 1)

        model_, batch_num, epochs = self._get_model_func_by_name(model_name)
        model = model_()

        model.compile(loss=K.losses.categorical_crossentropy,
                      optimizer=K.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        model.fit(self._train_img, self._train_label, 
                  batch_size=batch_num, 
                  epochs=epochs,
                  verbose=1,
                  validation_data=(self._test_img, self._test_label))

        score = model.evaluate(self._test_img, self._test_label, verbose=0)

        print('Test loss: ', score[0])
        print('Test accuracy: ', score[1])

        model.save_weights(os.path.join(self._data_folder, model_name + '.h5'))
        with open(os.path.join(self._data_folder, model_name + ".json"), 'w+') as jf:
            jf.write(model.to_json())

        self._model = model
        return self._model

    def _get_model_func_by_name(self, model_name='simple_cnn'):
        '''
        '''
        return self._models[model_name]

    def mnist_cnn(self):
        ''' acc: 0.53
        follow https://keras.io/examples/mnist_cnn/
        '''

        model = Sequential()
        model.add(Conv2D(32,
                         kernel_size=(3,3),
                         activation='relu',
                         input_shape=self._input_shape))
        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self._classes_num, activation='softmax'))
        
        return model

    def simple_cnn(self):
        ''' acc: 0.60
        '''

        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=self._input_shape))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(1024, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self._classes_num, activation='softmax'))
        
        return model

    def vgg16(self):
        '''
        '''
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=self._input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(self._classes_num, activation='softmax'))

        return model
