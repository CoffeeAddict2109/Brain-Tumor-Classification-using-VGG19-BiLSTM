import os
from flask import Flask, render_template, request, jsonify, url_for, send_file, send_from_directory
from keras.utils import load_img
from keras.utils import img_to_array
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import shutil

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './img/'

model = tf.keras.models.load_model('./model_vgg19+bilstm_skripsi.h5')
opt = Adam(learning_rate=0.000016, beta_1=0.91, beta_2=0.9994, epsilon=1e-08)
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

target_names = ['Glioma Tumor','Meningioma Tumor','Normal', 'Pituitary Tumor']

def is_grey_scale(img_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i,j))
            if r != g != b: 
                return False
    return True

@app.route('/', methods = ['GET'])
def tumor_detection():
    return render_template("index.html")

@app.route('/', methods = ['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224,224))
    if (is_grey_scale(image_path) == True):
        image1 = img_to_array(image)
        image1 = np.expand_dims(image1, axis=0)
        image1 = np.vstack([image1])
        prediksi = model.predict(image1)
        skor = np.max(prediksi)
        print(skor)
        classes = np.argmax(prediksi)
        hasil = target_names[classes]        
    else:
        hasil = 'Gambar tidak terdeteksi sebagai citra MRI'
    return render_template("hasil.html", result=hasil, img=imagefile.filename)

@app.route('/img/<fileimg>')
def send_uploaded_image(fileimg=''):
    return send_from_directory( app.config['UPLOAD_FOLDER'], fileimg)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
