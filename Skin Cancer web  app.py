import streamlit as stmlit
import numpy as np
import pandas as pd
import keras
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras import backend as K
import os
import time
import io
from PIL import Image
import tensorflow as tf
import plotly.express as px
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy


MODELSPATH = 'C:/Users'
DATAPATH = 'C:/Users/archive/HAM10000_images_part_1/'


def render_header():
    stmlit.write("""
        <p align="center"> 
            <H1> Skin cancer Analyzer 
        </p>

    """, unsafe_allow_html=True)


@stmlit.cache
def load_mekd():
    img = Image.open(DATAPATH+'/ISIC_0026893.jpg')
    return img


@stmlit.cache
def preprocess(image):
  
  image = Image.open(image)
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.efficientnet.preprocess_input(image)
  image = image[None, ...]
  return image

def best3_acc(y_actual, y_predict_value):
    return top_k_categorical_accuracy(y_actual, y_predict_value, k=3)

def best2_acc(y_actual, y_predict_value):
    return top_k_categorical_accuracy(y_actual, y_predict_value, k=2)

def load_models():

    model = load_model(MODELSPATH + 'modeltensorefficentb4.h5', 
                       custom_objects={"top_2_accuracy": best3_acc,
                                       "top_3_accuracy":best2_acc})
    return model


@stmlit.cache
def predict(x_test, model):
    Y_pred = model.predict(x_test)
    y_new = model.predict(x_test)
    K.clear_session()
    y_new = np.round(y_new, 2)
    y_new = y_new*100
    y_latest = y_new[0].tolist()
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    K.clear_session()
    return y_latest, Y_pred_classes


@stmlit.cache
def display_prediction(y_new):
    """Display image and preditions from model"""

    result = pd.DataFrame({'Probability': y_new}, index=np.arange(7))
    result = result.reset_index()
    result.columns = ['Classes', 'Probability']
    lesion_type_dict = {2: 'Benign keratosis-like lesions', 5: 'Melanocytic nevi', 3: 'Dermatofibroma',
                        4: 'Melanoma', 6: 'Vascular lesions', 1: 'Basal cell carcinoma', 0: 'Actinic keratoses'}
    result["Classes"] = result["Classes"].map(lesion_type_dict)
    return result


def main():
    stmlit.sidebar.header('Skin cancer Analyzer')
    stmlit.sidebar.subheader('Choose a page to proceed:')
    page = stmlit.sidebar.selectbox("", ["Sample Data", "Upload Your Image"])

    if page == "Sample Data":
        stmlit.header("Sample Data Prediction for Skin Cancer")
        stmlit.markdown("""
        **Now, this is probably why you came here. Let's get you some Predictions**

        You need to choose Sample Data
        """)

        mov_base = ['Sample Data I']
        movies_chosen = stmlit.multiselect('Choose Sample Data', mov_base)

        if len(movies_chosen) > 1:
            stmlit.error('Please select Sample Data')
        if len(movies_chosen) == 1:
            stmlit.success("You have selected Sample Data")
        else:
            stmlit.info('Please select Sample Data')

        if len(movies_chosen) == 1:
            if stmlit.checkbox('Show Sample Data'):
                stmlit.info("Showing Sample data---->>>")
                image = load_mekd()
                stmlit.image(image, caption='Sample Data', use_column_width=True)
                stmlit.subheader("Choose Training Algorithm!")
                if stmlit.checkbox('Click to fit Keras Model'):
                    model = load_models()
                    stmlit.success("Hooray !! Keras Model Loaded!")
                    if stmlit.checkbox('Show Prediction Probablity on Sample Data'):
                        x_test = preprocess(DATAPATH + '/ISIC_0026350.jpg')
                        y_new, Y_pred_classes = predict(x_test, model)
                        result = display_prediction(y_new)
                        stmlit.write(result)
                        if stmlit.checkbox('Display Probability Graph'):
                            fig = px.bar(result, x="Classes",
                                     y="Probability", color='Classes')
                            stmlit.plotly_chart(fig, use_container_width=True)

    if page == "Upload Your Image":

        stmlit.header("Upload Your Image")

        file_pth = stmlit.file_uploader('Upload an image', type=['png', 'jpg'])

        if file_pth is not None:
            x_test = preprocess(file_pth)
            img = Image.open(file_pth)
            img_array = np.array(image)

            stmlit.success('File Upload Success!!')
        else:
            stmlit.info('Please upload Image file')

        if stmlit.checkbox('Show Uploaded Image'):
            stmlit.info("Showing Uploaded Image ---->>>")
            stmlit.image(img_array, caption='Uploaded Image',
                     use_column_width=True)
            stmlit.subheader("Choose Training Algorithm!")
            if stmlit.checkbox('Keras'):
                model = load_models()
                stmlit.success("Hooray !! Keras Model Loaded!")
                if stmlit.checkbox('Show Prediction Probablity for Uploaded Image'):
                    y_new, Y_pred_classes = predict(x_test, model)
                    result = display_prediction(y_new)
                    stmlit.write(result)
                    if stmlit.checkbox('Display Probability Graph'):
                        fig = px.bar(result, x="Classes",
                                     y="Probability", color='Classes')
                        stmlit.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
