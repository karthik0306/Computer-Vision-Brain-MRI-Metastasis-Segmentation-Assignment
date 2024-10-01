import streamlit as st
import requests
import numpy as np
import cv2

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Segment"):
        response = requests.post("http://localhost:8000/predict/", files={"file": uploaded_file})
        prediction = response.json()["prediction"]
        st.image(np.array(prediction).reshape((128, 128)), caption='Segmented Output', use_column_width=True)
