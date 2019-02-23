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
            'simple_cnn' : (self.simple_cnn, 256, 30),
            'vgg16'     : (self.vgg16, 128, 12),
        }

        self._model = None

    def load_model(self, model_name='simple_cnn', data_folder='data', dataset=None):
        if None == dataset:
            self.load_saved_model(model_name, data_folder)
        else:
            self.make_model(dataset, model_name, data_folder)

    def load_saved_model(self, model_name='simple_cnn', data_folder='data'):
        '''
        '''
        with open(os.path.join(data_folder, model_name + '.json'), 'r') as jf:
            jm = jf.read()

        model = model_from_json(jm)
        model.load_weights(os.path.join(data_folder, model_name + ".h5"))

        model.compile(loss=K.losses.categorical_crossentropy,
                      optimizer=K.optimizers.Adadelta(),
                      metrics=['accuracy'])

        self._model = model
        return self._model

    def predict_image(self, img_file, model=None, target_width=48, target_height=48):
        img = image.load_img(img_file, color_mode='grayscale', target_size=(target_width, target_height))
        self.predict(image.img_to_array(img), model)
        
    def predict(self, img_array, model=None):
        if None == model:
            model = self._model
        if None == model:
            model = self.load_saved_model()

        img_arr = np.expand_dims(img_array, axis=0)

        properbilities = model.predict(img_arr)
        predictions = [p for p in properbilities]
        print(predictions)

    def make_model(self, dataset, model_name='simple_cnn', data_folder='data'):
        '''
        '''
        train_labels, train_images = dataset['train_data']
        test_labels, test_images = dataset['test_data']
        val_labels, val_images = dataset['val_data']
        classes_num = dataset['classes_num']
        input_shape = dataset['input_shape']

        if dataset['channels_first']:
            bk.set_image_data_format('channels_frist')
        else:
            bk.set_image_data_format('channels_last')

        model_, batch_num, epochs = self._get_model_func_by_name(model_name)
        model = model_(classes_num, input_shape)

        model.compile(loss=K.losses.categorical_crossentropy,
                      optimizer=K.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        model.fit(train_images, train_labels, 
                  batch_size=batch_num, 
                  epochs=epochs,
                  verbose=1,
                  validation_data=(val_images, val_labels))

        score = model.evaluate(test_images, test_labels, verbose=0)

        print('Test loss: ', score[0])
        print('Test accuracy: ', score[1])

        model.save_weights(os.path.join(data_folder, model_name + '.h5'))
        with open(os.path.join(data_folder, model_name + ".json"), 'w+') as jf:
            jf.write(model.to_json())

        self._model = model
        return self._model

    def _get_model_func_by_name(self, model_name='simple_cnn'):
        '''
        '''
        return self._models[model_name]

    def simple_cnn(self, classes_num, input_shape):
        '''
        '''
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
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
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(classes_num, activation='softmax'))
        
        return model

    def vgg16(self, classes_num, input_shape):
        '''
        '''
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
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
        model.add(Dense(classes_num, activation='softmax'))

        return model
