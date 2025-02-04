import cv2 as cv
from rembg import remove


def RemoveBackground(image, name, ekstensi):
    img = cv.imread(image)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = remove(img)
    nameFile = "./static/img_proses/" + name + "." + ekstensi + ""
    cv.imwrite(nameFile, img)
    return nameFile
