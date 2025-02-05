from flask import Flask, request, render_template, url_for
import os

import glob
import numpy as np
import cv2
from rembg import remove
from flask import Flask, request, render_template, url_for
from pathlib import Path
import resizeRBg as rb
import featureEx as fe
import hEqualization
import knn as klasifikasi

app = Flask(__name__,template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = './static/'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.form.get('images')
    img_path = os.path.join(app.config['UPLOAD_FOLDER'],file)
    # image_path = "./static/"+file
    filename = img_path
    image = cv2.imread(filename)

    # Tentukan ukuran yang diinginkan
    width = 512
    height = 512

    # Resize citra dengan ukuran baru
    resized_image = cv2.resize(image, (width, height))

    #hapus background
    output = remove(resized_image)

    nameFile = "./static/img_process/rembg.png"
    cv2.imwrite(nameFile, output)

    img_he = hEqualization.HistogramEqualization(output)
    img_he = glob.glob(img_he)

    data = fe.Feature(img_he, 'HSV')

    hasil = klasifikasi.hasil(data)

    print(hasil)
    if (hasil[0] == 'jahe' or hasil[0] == 'kencur' or hasil[0] == 'kunyit' or hasil[0] == 'lengkuas'):
        rembg = "static/img_proses/he.png"
        he = "static/img_proses/rembg.jpg"
        hsv = "static/img_proses/hsv.jpg"
        glcm = "static/img_proses/gray.jpg"
        image = "static/images/"+file
        return render_template('index.html', hasil='{}'.format(hasil[0]), rb='{}'.format(rembg), clahe='{}'.format(he), hsv='{}'.format(hsv), gray='{}'.format(glcm), image='{}'.format(image), h='{}'.format(data[0][0]), s='{}'.format(data[0][1]),
                               v='{}'.format(data[0][2]), con0='{}'.format(data[0][3]), con45='{}'.format(data[0][4]), con90='{}'.format(data[0][5]), con135='{}'.format(data[0][6]), hom0='{}'.format(data[0][7]), hom45='{}'.format(data[0][8]),
                               hom90='{}'.format(data[0][9]), hom135='{}'.format(data[0][10]), enr0='{}'.format(data[0][11]), enr45='{}'.format(data[0][12]), enr90='{}'.format(data[0][13]), enr135='{}'.format(data[0][14]),
                               corr0='{}'.format(data[0][15]), corr45='{}'.format(data[0][16]), corr90='{}'.format(data[0][17]), corr135='{}'.format(data[0][18]))

if __name__ == '__main__':
    app.run(debug=True)