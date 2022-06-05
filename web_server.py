from flask import Flask, session, request, send_file
import json
import sys
import os
from sys import platform
from envReader import getValue
from utils import base64ToArray

def initServer():
    app = Flask('Themistoklis')

    @app.route('/', methos=['POST', 'GET', 'DELETE'])
    def main():
        if request.method == 'GET':
            return 'Hello World!'
        elif request.method == 'POST':
            return json.dumps({"status": "not_supported"})
        elif request.method == 'DELETE':
            return json.dumps({"status": "not_supported"})

    @app.route('/detect', methods=['GET', 'POST'])
    def detect():
        if request.method == 'GET':
            return json.dumps({"status": "not_supported"})
        elif request.method == 'POST':
            data = json.loads(request.data)
            image_data = data['image']

            img_array = base64ToArray(image_data)

            return json.dumps({"status": "success"})

    app.run(host=getValue("WEBHOST_IP"), port=getValue("WEBHOST_PORT"))