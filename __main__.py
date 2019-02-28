import ferplus as fp
import datetime

if __name__ == "__main__":
    model = fp.load_model()

    print(datetime.datetime.now())

    print(fp.predict(r'e:\\temp\\a.jpg', model=model))
    print(fp.predict(r'e:\\temp\\b.png', model=model))
    print(fp.predict(r'e:\\temp\\c.png', model=model))
    print(fp.predict(r'e:\\temp\\d.jpg', model=model))
    print(fp.predict(r'e:\\temp\\e.jpg', model=model))

    print(datetime.datetime.now())