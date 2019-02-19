import os
import csv
import numpy as np
from PIL import Image

class FerPlus:
    '''
    '''

    def __init__(self):
        self.IMG_WIDTH = 48
        self.IMG_HEIGHT = 48

    def extract_imgs(self, fer_csv='fer\\fer2013.csv', img_folder='fer+\\img'):
        _fer_csv = fer_csv
        _fer_p_folder = img_folder

        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

        with open(_fer_csv, mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            idx = 0
            for row in reader:
                parr = list(map(float, row[1].split(' ')))
                pixel = np.asarray(parr).reshape(self.IMG_HEIGHT, self.IMG_WIDTH)
                im = Image.fromarray(pixel).convert('L')
                imgpath = os.path.join(_fer_p_folder, 'fer' + '{0:0>7}'.format(idx) + '.png')
                im.save(imgpath)
                idx += 1
                