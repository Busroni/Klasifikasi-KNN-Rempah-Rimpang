import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern

# Fungsi untuk menghitung GLCM dan mengambil fitur
def Features(image):

    # Ekstrak komponen warna dari citra RGB <<<RGB
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    # Hitung rata-rata RGB
    average_red = np.mean(red)
    average_green = np.mean(green)
    average_blue = np.mean(blue)

    print (average_red, average_green, average_blue)

    rgb = [average_red, average_green, average_blue]
    # RGB <<<

    # HSV <<< # Konversi gambar ke ruang warna HSL
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # # # tes save file -------------------------
    nameFile = "./static/img_process/hsv.png"
    cv2.imwrite(nameFile, hsv_image)
    # # # #---------------------------------------

    # hsv adalah ruang warna tiga saluran, akses setiap saluran
    hue = hsv_image[:, :, 0]
    lightness = hsv_image[:, :, 1]
    value = hsv_image[:, :, 2]

    # Hitung nilai rata-rata
    average_hue = np.mean(hue)
    average_lightness = np.mean(lightness)
    average_value = np.mean(value)
    
    hsv = [average_hue,average_lightness,average_value]

    # HSL <<<

    # Konversi ke grayscale <<< GLCM
    gray = cv2.cvtColor(image,cv2.COLOR_RGBA2GRAY)

    # # # tes save file -------------------------
    nameFile = "./static/img_process/gray.png"
    cv2.imwrite(nameFile, gray)
    # # # #---------------------------------------

    # Ekstraksi fitur LBP
    radius = 5
    n_points = 24 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp = lbp.astype(np.uint8)

    # Hitung matriks GLCM dengan semua sudut
    glcm = graycomatrix(lbp, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

    # Ekstraksi fitur dari matriks GLCM
    contrast = graycoprops(glcm, 'contrast').ravel()
    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
    homogeneity = graycoprops(glcm, 'homogeneity').ravel()
    energy = graycoprops(glcm, 'energy').ravel()
    correlation = graycoprops(glcm, 'correlation').ravel()

    # Gabungkan fitur ke dalam satu baris DataFrame
    features = np.concatenate([contrast,dissimilarity,homogeneity, energy, correlation, rgb, hsv])

    features = features.reshape(1, -1)

    return features