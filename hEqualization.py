import cv2
import numpy as np

def HistogramEqualization(image):
# --Load citra dengan latar belakang transparan
    image_process = image

    # Ambil saluran alpha (mask) dari citra
    alpha = image[:, :, 3]

    # Hitung histogram citra sebelum penyejajaran
    hist, bins = np.histogram(image_process.flatten(), 256, [0, 256])

    # Hitung fungsi distribusi kumulatif (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Lakukan penyejajaran histogram pada citra grayscale
    # equalized_image = cv2.equalizeHist(image_process)
    equalized_image_red = cv2.equalizeHist(image_process[:,:,0])
    equalized_image_green = cv2.equalizeHist(image_process[:,:,1])
    equalized_image_blue = cv2.equalizeHist(image_process[:,:,2])

    # Gabungkan kembali saluran grayscale dan alpha menjadi citra RGBA
    equalized_image_rgba = np.zeros_like(image_process)
    equalized_image_rgba[:, :, 0] = equalized_image_red
    equalized_image_rgba[:, :, 1] = equalized_image_green
    equalized_image_rgba[:, :, 2] = equalized_image_blue
    equalized_image_rgba[:, :, 3] = alpha
    
    # write image
    nameFile = "./static/img_process/he.png"
    cv2.imwrite(nameFile, equalized_image_rgba)
    
    return equalized_image_rgba
