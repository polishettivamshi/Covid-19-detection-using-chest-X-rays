import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras import preprocessing
import time

# function to predict the uploaded image
def predict(image, classifier):
    image_shape = (256, 256)  # shape of the uploaded image
    test_image = image.resize(image_shape)  # reshaping the uploaded image
    
    # Convert to grayscale if needed (remove if your model expects RGB)
    if test_image.mode != 'L':
        test_image = test_image.convert('L')
    
    test_image = preprocessing.image.img_to_array(test_image)  # converting to numpy array
    test_image = test_image / 255.0  # normalizing
    
    # Add channel dimension (will be 1 for grayscale)
    test_image = np.expand_dims(test_image, axis=-1)
    
    # Add batch dimension
    test_image = np.expand_dims(test_image, axis=0)

    class_names = ['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia']
    predictions = classifier.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    # print(f"scores:{scores}")
    confidence = (100 * np.max(scores)).round(2) * 2
    result = f"The uploaded image belongs to {class_names[np.argmax(scores)]} with a {confidence} % confidence."

    return result

# main function

def main():
    st.header('Covid 19 detection using X-Ray images')
    st.markdown('This project takes X-Ray images as input and predicts whether it belong to the following classes')
    st.markdown('Covid-19')
    st.markdown('Healthy')
    st.markdown('Lung Opacity')
    st.markdown('Viral Pneumonia')

    # loading the saved model
    model = load_model('model.h5', compile=False)

    # uploading the image
    uploaded_image = st.file_uploader("Choose an image to be predicted", type=["png","jpg","jpeg"])

    if uploaded_image is not None:
        image_data = Image.open(uploaded_image)
        
        # Display image with fixed width (e.g., 500 pixels)
        st.image(image_data, 
                caption='Uploaded X-Ray image', 
                width=500,  # Set your desired fixed width
                use_column_width=False)  # Important to keep this False

        class_btn = st.button('Classify')
        
        if class_btn:
            if uploaded_image is None:
                st.write('Please upload a valid image')
            else:
                with st.spinner("Classifying..."):
                    plt.imshow(image_data)
                    plt.axis("off")
                    predictions = predict(image_data, model)
                    time.sleep(1)
                    st.success('Classified')
                    st.write(predictions)

if(__name__ == '__main__'):
    main()