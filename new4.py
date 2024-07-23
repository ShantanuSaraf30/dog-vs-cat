import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Load the trained model
cnn = tf.keras.models.load_model('cats_dogs_model_new.keras')

# Define a class for processing video frames
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.result = None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        img = Image.fromarray(img)

        # Preprocess the image
        img = img.resize((64, 64))
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        result = cnn.predict(img_array)
        self.result = 'Dog' if result[0][0] > 0.5 else 'Cat'
        
        return img_array[0]

st.title("Dog vs Cat Detection")
st.write("Use the camera to capture an image and predict if it's a dog or a cat.")

ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

if ctx.video_processor:
    st.write(f"Prediction: {ctx.video_processor.result}")
