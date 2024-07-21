#!/usr/bin/env python
'''
Name: NeuralCraft
Github: https://github.com/aqmeraamir

This is an API for the Minecraft plugin i'm developing that allows me to connect the neural network i've made on Python, 
with my Java code.
'''

# Import necessary libraries
from flask import Flask, request, jsonify
import digit_recognisor
import numpy as np

# Create the app
app = Flask(__name__)

@app.route('/predict_digit', methods=['POST'])
def predict_digit():

    try:
        # retrieve the array from the request
        data = request.get_json()
        array = data["array"]
    
    except: 
        return jsonify({"error": "Missing data for array"}), 400
    
    
    try:
        # reshape the array into the correct format for the model
        array = array.split(',')
        array = np.array(array)
        array = array.reshape(1, 28, 28)
        array = array.astype(float)
       
    except:
        return jsonify({"error": "invalid syntax given for the array"}), 400 
    
    
    try:
        # feed the array into the model to retrieve a prediction
        prediction = digit_recognisor.model.predict(array)
    except:
        return jsonify({"error": "failed to process data in the model"}), 400 
    

    # formatting and returning the response
    response = {
        "activations": str(prediction.squeeze()),
        "most_likely": str(np.argmax(prediction))
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(port=8080) 

