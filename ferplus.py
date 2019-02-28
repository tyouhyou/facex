import os
import csv
import numpy as np
import pickle
from PIL import Image
from kmodel import KModel as km

IMG_WIDTH = 48
IMG_HEIGHT = 48
CHANNEL = 1

def load_model(model_data_folder='data', model_data_name='fer+_simple_cnn'):
    '''
    make model for facial express
    '''
    m = km()
    model = None

    if (os.path.exists(os.path.join(model_data_folder, model_data_name + '.h5')) and
        os.path.exists(os.path.join(model_data_folder, model_data_name + '.json'))):
        model = m.load_model(data_name=model_data_name)
    else:
        dataset = load_data()
        model = km().make_model(dataset, data_name='fer+_simple_cnn')

    return model

def predict(data, width=48, height=48, model=None):
    if type(data) is str:
        result = km().predict_image(data, model=model, target_height=48, target_width=48)
    else:
        result = km().predict(data, model)

    return result

def extract_imgs(fer_csv='fer\\fer2013.csv', img_folder='fer+\\img'):
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    with open(fer_csv, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        idx = 0
        for row in reader:
            parr = list(map(float, row[1].split(' ')))
            pixel = np.asarray(parr).reshape(IMG_HEIGHT, IMG_WIDTH)
            im = Image.fromarray(pixel).convert('L')
            imgpath = os.path.join(img_folder, 'fer' + '{0:0>7}'.format(idx) + '.png')
            im.save(imgpath)
            idx += 1

def load_data(data_file = 'fer+\\data\\fer+.data'):
    if not os.path.exists(data_file):
        return load_ferp_data()
    else :
        return load_pickle_data(data_file)

def load_ferp_data(fernewcsv=r'fer+\\fer2013new.csv', img_folder=r'fer+\\img', 
                   pickle_file=r'fer+\\data\\fer+.data', save_data=True):
    with open(fernewcsv, mode='r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        
        classes_num = 9
        train_labels = []
        train_images = []
        test_labels = []
        test_images = []
        val_labels = []
        val_images = []
        for row in reader:
            if (10 == row[11]): continue

            classes = np.asarray([row[i+2] for i in range(classes_num)], dtype='float32')
            imgfile = os.path.join(img_folder, row[1])
            pixels = (np.asarray(Image.open(imgfile), dtype='float32') / 255).reshape(IMG_HEIGHT,
                                                                                      IMG_WIDTH,
                                                                                      CHANNEL)

            if 'Training' == row[0]:
                train_labels.append(classes)
                train_images.append(pixels)
            elif 'PublicTest' == row[0]:
                val_labels.append(classes)
                val_images.append(pixels)
            elif 'PrivateTest' == row[0]:
                test_labels.append(classes)
                test_images.append(pixels)

        ferp_data = {
            'channels_first': False,
            'classes_num': classes_num,
            'input_shape': (IMG_WIDTH, IMG_HEIGHT, CHANNEL),
            'train_data': (np.asarray(train_labels), np.asarray(train_images)),
            'test_data': (np.asarray(test_labels), np.asarray(test_images)),
            'val_data' : (np.asarray(val_labels), np.asarray(val_images))
        }

        if save_data:
            df = os.path.dirname(pickle_file)
            if not os.path.exists(df):
                os.makedirs(df)

            with open(pickle_file, 'wb+') as pf:
                pickle.dump(ferp_data, pf)

        return ferp_data

def load_pickle_data(pickle_file='fer+\\data\\fer+.data'):
    with open(pickle_file, 'rb') as pf:
        pickle_data = pickle.load(pf)

    return pickle_data