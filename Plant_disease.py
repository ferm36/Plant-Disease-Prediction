import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Loading model
model = load_model("/Users/fernando/PycharmProjects/Plant_Disease/Data/plant_disease.h5")

# Names of classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Setting title of Application
st.title("Detección de enfermedades en plantas")
st.markdown("Sube una imagen de una hoja de la planta")

# Uploading the image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button("Predecir")

if submit:
    if plant_image is not None:
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)

        # Resize Image
        opencv_image = cv2.resize(opencv_image, (256, 256))

        # Convert image to 4 Dimension
        opencv_image.shape = (1, 256, 256, 3)

        # Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("Este es "+result.split('-')[0]+" hoja con "+ result.split('-')[1]))