import cv2
import pathlib
from rembg import remove

def RemoveBackground(image2, name):

    # Baca gambar menggunakan OpenCV
    # image2 = cv2.imread(image)

    # Tentukan ukuran yang diinginkan
    width = 512
    height = 512

    # Resize citra dengan ukuran baru
    resized_image = cv2.resize(image2, (width, height))

    #hapus background
    output = remove(resized_image)

    nameFile = "./static/img_process/" + name + ".png"
    cv2.imwrite(nameFile, output)
    return nameFile