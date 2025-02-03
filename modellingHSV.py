import cv2 as cv
import numpy as np

class ModellingHSV:
    def __init__(self, image):
        self.image = image

    def process(self,):
        arr = []
        hasilHSV = []

        readImage = cv.imread(self.image)
        readImage = cv.cvtColor(readImage, cv.COLOR_BGR2RGB)
        normalisasi = readImage.astype(np.float32)/255.0
        hsv_h = np.zeros_like(normalisasi[:, :, 0])
        hsv_s = np.zeros_like(normalisasi[:, :, 0])
        hsv_v = np.zeros_like(normalisasi[:, :, 0])
        for i in range(normalisasi.shape[0]):
            for j in range(normalisasi.shape[1]):
                Rnor, Gnor, Bnor = normalisasi[i][j]

                max_value = max(Rnor, Gnor, Bnor)
                min_value = min(Rnor, Gnor, Bnor)

                value = max_value
                    # Calculate the Saturation
                if max_value == 0:
                    saturation = 0
                else:
                    saturation = 1 - (min_value / max_value)

                delta = max_value - min_value

                    # Calculate the Hue
                if saturation == 0:
                    hue = 0
                elif max_value == Rnor:
                    hue = 60 * ((Gnor - Bnor) / delta)
                elif max_value == Gnor:
                    hue = (60 * ((Bnor - Rnor) / delta) + 2)
                else:
                    hue = (60 * ((Rnor - Gnor) / delta) + 4)

                if hue < 0:
                    hue = hue + 360


                hsv_h[i][j] = hue
                hsv_s[i][j] = saturation
                hsv_v[i][j] = value

        hsv_h = (hsv_h / 2)
        hsv_s = hsv_s * 255
        hsv_v = hsv_v * 255

        hasilHSV = cv.merge((hsv_h, hsv_s, hsv_v))

        hsv_array = np.array(hasilHSV)
        mean_hsv = np.mean(hsv_array, axis=(0, 1))
        h, s, v = mean_hsv[0], mean_hsv[1], mean_hsv[2]
        arr = np.column_stack(( h, s, v))
        print(arr)
            # print(arr)
            # hasilHSV = remove(hasilHSV)
        newNameImage = './static/img_proses/hsv.jpg'
        cv.imwrite(newNameImage, hasilHSV)
        print("proses "+newNameImage + " selesai")
        return arr
