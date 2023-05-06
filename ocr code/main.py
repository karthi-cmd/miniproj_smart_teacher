# Importing the Keras libraries and packages
from keras.models import load_model
model = load_model('/content/drive/MyDrive/Colab Notebooks/mnistCNN.h5')

from PIL import Image
import numpy as np


img = Image.open('/content/drive/MyDrive/Colab Notebooks/sample input 5.png').convert("L")
img = img_resized #.resize((28,28))
im2arr = np.array(img)
im2arr = im2arr.reshape(1,28,28,1)
# Predicting the Test set results
y_pred = model.predict(im2arr)
print(y_pred)
print(np.argmax(y_pred))
