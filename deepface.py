from deepface import Deepface
import numpy as np
from PIL import Image

image_file_path = "C:://deepface/Sample/Sample/01.원천데이터/0001/예측모델1.jpg"

img = Image.open(image_file_path)
img = np.array

result = Deepface.analyze(img, actions=['age'], enforce_detection = False)

predicted_age = result[0]["age"]

print("예측나이: ", predicted_age)