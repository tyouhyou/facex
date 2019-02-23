from facialexpress import FacialExpress as fe
import ferplus as fp
import datetime

if __name__ == "__main__":
    # fp.extract_imgs()

    fer = fe()
    # dataset = fp.load_data()
    fer.load_model(dataset=None)

    print(datetime.datetime.now())

    fer.predict_image(r'e:\\temp\\a.jpg')
    fer.predict_image(r'e:\\temp\\b.png')
    fer.predict_image(r'e:\\temp\\c.png')
    fer.predict_image(r'e:\\temp\\d.jpg')
    fer.predict_image(r'e:\\temp\\e.jpg')

    print(datetime.datetime.now())