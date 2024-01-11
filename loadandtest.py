import keras
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the model
model = keras.models.load_model('/content/drive/MyDrive/cnn.h5')

# Load the image
img = load_img('/content/drive/MyDrive/tomato/val/Tomato__Spider_mites Two-spotted_spider_mite/00fa99e8-2605-4d72-be69-98277587d84b__Com.G_SpM_FL 1453.JPG')

# Preprocess the image
img = img.resize((224, 224))
img = img_to_array(img)
img = img / 255.0

# Add a new dimension to the image
img = img[np.newaxis, ...]

# Make a prediction
prediction = model.predict(img)

# Get the class names
class_names = ['Tomato__Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites Two-spotted_spider_mite','Tomato_Target_Spot','Tomato_Tomato_Yellow_Leaf_Curl_Virus','Tomato_Tomato_mosaic_virus','Tomato__healthy']
class_names = np.array(class_names)[np.argmax(prediction, axis=1)]

# Print the class names
print(class_names)