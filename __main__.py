import os
import ferplus as fp
import datetime

if __name__ == '__main__':

    test_folder = 'E:\\test_images\\faceonly'
    model = fp.load_model()

    print(datetime.datetime.now())

    for f in os.listdir(test_folder):
        imgf = os.path.join(test_folder, f)
        if (os.path.isfile(imgf) and imgf.lower().endswith(('.jpg', '.jpeg', '.png'))):
            print('{0}-> {1}'.format(f, fp.predict(imgf, model=model)))

    print(datetime.datetime.now())