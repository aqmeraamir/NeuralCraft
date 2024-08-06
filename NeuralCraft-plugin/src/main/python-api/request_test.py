import requests
import numpy as np
import cv2

url = "http://127.0.0.1:8080/predict_digit"

img_array = np.invert(np.array(cv2.imread('test.png')[:,:,0]).flatten())
array = [str(x) for x in img_array]
array = ','.join(array)


array = "3"
data = {"array": array}

response = requests.post(url)
print(response.json())
