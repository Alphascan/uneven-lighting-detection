#this file has the flask server which hosts the algorithm and allows incoming connection

from flask import Flask, request, redirect, render_template, jsonify
import io
import numpy as np
import cv2
from detector import Detector
from flask_cors import CORS
import boto3
import yaml

app = Flask(__name__)
CORS(app)

#there are two ways to upload the photo

#the first way is to send the picture to an html page that we made for local testing purposes

@app.route('/')
def upload_photo():
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def inference():
    if request.method == 'POST' and 'photo' in request.files:
        photo = request.files['photo']
        in_memory_file = io.BytesIO()
        photo.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        img = cv2.imdecode(data, color_image_flag)
        d = Detector()
        output = d.process_image(img)

        print(output)
        return output

#this second way is what the application actually uses
#first it passes in the filename of the image from the S3 bucket
@app.route('/process-s3/<object_name>', methods=['GET'])
def inference_s3(object_name):

    with open('config.yml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    #goes to S3 and picks out that file
    s3 = boto3.client(
        's3',
        aws_access_key_id=config['s3']['aws_access_key_id'],
        aws_secret_access_key=config['s3']['aws_secret_access_key']
    )

    s3.download_file(config['s3']['bucket_name'], object_name, 'photo.jpg')
    img = cv2.imread('photo.jpg')
    d = Detector()
    output = d.process_image(img)

    print(output)
    return output

# start flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
