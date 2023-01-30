from flask import Flask, request, Response, jsonify, json, send_from_directory, abort
import os
from keras.models import load_model
import os
from keras.preprocessing import image
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.applications.vgg16 import preprocess_input, decode_predictions

import tensorflow as tf

import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS

classifier = load_model('equipment_detector.h5')

stored_dict = {0: '1_bench', 1: '2_dumbbell', 2: '4_leg-extension', 3: '3_leg-press'}
stored_dict_n = {0: '1_bench', 1: '2_dumbbell', 2: '4_leg-extension', 3: '3_leg-press'}
#gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#for device in gpu_devices:
   #tf.config.experimental.set_memory_growth(device, True)
config =  tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
app = Flask(__name__)
CORS(app)
#API that returns image with detections on it
@app.route('/image', methods= ['POST'])
def get_image():

    image_data = request.json['image']
    image_data = bytes(image_data, encoding="ascii")
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    im.save('image.jpg')

    image_path = 'image.jpg'
    img = image.load_img(image_path, target_size=(256,256))
    img = image.img_to_array(img)
    img = img/255 #convert to grayscale
    img = np.expand_dims(img, axis=0)

    result = classifier.predict(img, 1, verbose=0)

    class_predicted = np.argmax(result[0])
    print("RESULT==============================")
    print(class_predicted)
    
    response = stored_dict[class_predicted]
    print("response")
    print(response)

    #if result[0][class_predicted] <= 0.5:
	#    response = 'no hay coincidencias'

    #index_predict = np.argmax(pred[0])
    print("class=", json.dumps(stored_dict[class_predicted]))
    os.remove('image.jpg')
    try:
        return Response(response=json.dumps(response), status=200, mimetype='application/json')
    except FileNotFoundError:
        abort(404)

@app.route('/test', methods= ['GET'])
def test():
    try:
        return Response(response=json.dumps("works!!"), status=200, mimetype='application/json')
    except FileNotFoundError:
        abort(404)

if __name__ == '__main__':
    app.run(debug=True)
