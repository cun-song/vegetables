import matplotlib.pyplot as plt
import numpy as py
import os
import PIL
from flask import Flask, render_template,request
import numpy as py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from zipfile import ZipFile
import pickle
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
model_mobnet = load_model('mobnet.h5')
model_resnet = load_model('resnet.h5')
model_vgg16 = load_model('vgg16.h5')
class_names = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']
picfolder = os.path.join('static','images')
app.config['imagedir'] = picfolder

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    for f in os.listdir('static/images'):
        os.remove(os.path.join('static/images', f))
    imagefile = request.files['imagefile']
    img_height, img_width = 224,224
    image_path = "./static/images/" + imagefile.filename
    imagefile.save(image_path)
    saveimage = cv2.imread(str(image_path))
    image_resized = cv2.resize(saveimage, (img_height, img_width))    
    image = py.expand_dims(image_resized, axis=0)
    if (request.form.get('model')=='1'): 
        pred = model_mobnet.predict(image)
        output_class = class_names[py.argmax(pred)]
        pic = os.path.join(app.config['imagedir'],imagefile.filename)
        return render_template('index.html',prediction=output_class,uimage=pic)
    elif (request.form.get('model')=='2'): 
        pred = model_resnet.predict(image)
        output_class = class_names[py.argmax(pred)]
        pic = os.path.join(app.config['imagedir'],imagefile.filename)
        return render_template('index.html',prediction=output_class,uimage=pic)
    elif (request.form.get('model')=='3'): 
        pred = model_vgg16.predict(image)
        output_class = class_names[py.argmax(pred)]
        pic = os.path.join(app.config['imagedir'],imagefile.filename)
        return render_template('index.html',prediction=output_class,uimage=pic)
    return render_template('index.html',prediction='Error, Please Select Model!!')

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')