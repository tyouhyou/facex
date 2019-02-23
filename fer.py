import os
import csv
import pickle
import numpy as np

IMG_WIDTH = 48
IMG_HEIGHT = 48
CHANNEL = 1

def load_data(data_file=r'fer\\data\\fer.data'):
    if (os.path.exists(data_file)):
        return load_pickle_data(data_file)
    else:
        return load_fer_data()
    
def load_fer_data(fer_file=r'fer\\fer2013.csv', save_to_pickle=True, data_file=r'fer\\data\\fer.data'):
    train_label = []
    train_img = []
    test_label = []
    test_img = []
    val_label = []
    val_img = []
    with open(fer_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for row in reader:
            if (row[2] == 'Training'):
                train_label.append(int(row[0]))
                train_img.append(list(map(int, row[1].split(' '))))
            elif (row[2] == 'PublicTest'):
                test_label.append(int(row[0]))
                test_img.append(list(map(int, row[1].split(' '))))
            elif (row[2] == 'PrivateTest'):
                val_label.append(int(row[0]))
                val_img.append(list(map(int, row[1].split(' '))))

    fer_data = {
        'channels_first': False,
        'classes_num': 7,
        'input_shape': (IMG_WIDTH, IMG_HEIGHT, CHANNEL),
        'train_data' : (np.asarray(train_label, dtype='float32'), 
                        np.asarray([np.asarray(x, dtype='float32').reshape(IMG_WIDTH, IMG_HEIGHT, CHANNEL) for x in train_img])),
        'test_data'  : (np.asarray(test_label, dtype='float32'), 
                        np.asarray([np.asarray(x, dtype='float32').reshape(IMG_WIDTH, IMG_HEIGHT, CHANNEL) for x in test_img])),
        'val_data'   : (np.asarray(val_label, dtype='float32'), 
                        np.asarray([np.asarray(x, dtype='float32').reshape(IMG_WIDTH, IMG_HEIGHT, CHANNEL) for x in val_img]))
    }

    if save_to_pickle:
        with open(data_file, 'wb') as pk:
            pickle.dump(fer_data, pk)
    
    return fer_data

def load_pickle_data(data_file=r'fer\\data\\fer.data'):
    with open(data_file, 'rb') as pk:
        pickle_data = pickle.load(pk)

    return pickle_data