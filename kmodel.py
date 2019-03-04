import os
import pkgutil
if pkgutil.find_loader('cntk'): os.environ['KERAS_BACKEND'] = 'cntk'
import numpy as np
import h5py
import keras as K
from keras import backend as bk
from keras.models import load_model, model_from_json
from keras.preprocessing import image
import knm

class KModel:
    ''' 
    for building and using keras model
    '''
    
    def __init__(self):
        '''
        '''
        self._models = {
            'simple_cnn' : (knm.simple_cnn, 256, 30),
            'vgg16'     : (knm.vgg16, 1024, 30),
        }

        self._model = None

    def load_model(self, dataset=None, model_name='simple_cnn', data_folder='data', data_name=None):
        '''
        '''
        model = None
        if None == dataset:
            model = self.load_saved_model(model_name, data_folder, data_name)
        else:
            model = self.make_model(dataset, model_name=model_name, data_folder=data_folder, data_name=data_name)
        return model

    def load_saved_model(self, model_name='simple_cnn', data_folder='data', data_name=None):
        '''
        '''
        if None == data_name:
            data_name = model_name

        with open(os.path.join(data_folder, data_name + '.json'), 'r') as jf:
            jm = jf.read()

        model = model_from_json(jm)
        model.load_weights(os.path.join(data_folder, data_name + ".h5"))

        self._model = model
        return self._model

    def predict_image(self, img_file, target_width, target_height, model=None, 
                      color_mode='grayscale', channel_last=False):
        '''
        '''
        if channel_last:
            bk.set_image_data_format('channels_last')

        img = image.load_img(img_file, color_mode=color_mode, target_size=(target_width, target_height))
        return self.predict(image.img_to_array(img), model)
        
    def predict(self, img_array, model=None):
        '''
        '''
        if None == model:
            model = self._model
        if None == model:
            model = self.load_saved_model()
        
        img_arr = np.expand_dims(img_array, axis=0)

        properbilities = model.predict(img_arr)
        return properbilities[0]

    def make_model(self, dataset, data_folder='data', data_name=None, model_name='simple_cnn', 
                   loss_fun=None, optimizer=None, metrics='accuracy'):
        '''
        '''
        train_labels, train_images = dataset['train_data']
        test_labels, test_images = dataset['test_data']
        val_labels, val_images = dataset['val_data']
        classes_num = dataset['classes_num']
        input_shape = dataset['input_shape']

        if dataset['channels_first']:
            bk.set_image_data_format('channels_first')
        else:
            bk.set_image_data_format('channels_last')

        model_, batch_num, epochs = self._get_model_func_by_name(model_name)
        model, loss, opt = model_(classes_num, input_shape)

        if not None == loss_fun:
            loss = loss_fun

        if not None == optimizer:
            opt = optimizer
        
        model.compile(loss=loss,
                      optimizer=opt,
                      metrics=[metrics])
        
        model.fit(train_images, train_labels, 
                  batch_size=batch_num, 
                  epochs=epochs,
                  verbose=1,
                  validation_data=(val_images, val_labels))

        score = model.evaluate(test_images, test_labels, verbose=0)

        print('Test loss: ', score[0])
        print('Test accuracy: ', score[1])

        if None == data_name:
            data_name = model_name

        model.save_weights(os.path.join(data_folder, data_name + '.h5'))
        with open(os.path.join(data_folder, data_name + ".json"), 'w+') as jf:
            jf.write(model.to_json())

        self._model = model
        return self._model

    def _get_model_func_by_name(self, model_name='simple_cnn'):
        '''
        '''
        return self._models[model_name]

