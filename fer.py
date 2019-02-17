import os
import csv
import pickle
import numpy as np

class FerData:
    """ extract data for training, testing from fer or saved pickles
    """

    IMG_WIDTH = 48
    IMG_HEIGHT = 48
    CHANNELS = 1

    def __init__(self, ff=None, pf=None):
        self._fer_file = ff
        self._pickle_file = pf

        if None == self._fer_file:
            self._fer_file = os.path.join("fer", "fer2013.csv")
        
        if None == self._pickle_file:
            if not os.path.exists('data'):
                os.makedirs('data')
                
            self._pickle_file = os.path.join("data", "fer.data")

    def load_data(self):
        if (os.path.exists(self._pickle_file)):
            return self.load_pickle_data()
        else:
            return self.load_fer_data()
        
    def load_fer_data(self, save_to_pickle=True):
        train_label = []
        train_img = []
        test_label = []
        test_img = []
        with open(self._fer_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader, None)
            for row in reader:
                if (row[2] == "Training"):
                    train_label.append(int(row[0]))
                    train_img.append(list(map(int, row[1].split(' '))))
                elif (row[2] == "PrivateTest"):
                    test_label.append(int(row[0]))
                    test_img.append(list(map(int, row[1].split(' '))))

        fer_data = (
            (
                np.asarray(train_label, dtype=int), 
                np.asarray([np.asarray(x, dtype=int).reshape(FerData.IMG_WIDTH,
                                                             FerData.IMG_HEIGHT,
                                                             FerData.CHANNELS)
                            for x in train_img])
            ), 
            (
                np.asarray(test_label, dtype=int), 
                np.asarray([np.asarray(x, dtype=int).reshape(FerData.IMG_WIDTH,
                                                             FerData.IMG_HEIGHT,
                                                             FerData.CHANNELS)
                            for x in test_img])
            ))

        if save_to_pickle:
            with open(self._pickle_file, 'wb') as pk:
                pickle.dump(fer_data, pk)
        
        return fer_data

    def load_pickle_data(self):
        with open(self._pickle_file, 'rb') as pk:
            pickle_data = pickle.load(pk)

        return pickle_data