import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import time

# Load the trained model once at the start
cnn = tf.keras.models.load_model('a.keras')

# Define a class for processing video frames
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.result = None
        self.last_frame_time = 0  # Track time to reduce redundant predictions
        self.frame_skip = 5  # Process 1 out of every 5 frames for optimization
    
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        # Only process every Nth frame based on frame_skip to reduce workload
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_skip:
            return np.zeros((64, 64, 3), dtype=np.uint8)  # Return a blank frame

        self.last_frame_time = current_time
        img = frame.to_ndarray(format="bgr24")
        img = Image.fromarray(img)

        # Preprocess the image
        img = img.resize((64, 64))
        img_array = np.array(img) / 255.0  # Normalize between 0 and 1
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = cnn.predict(img_array, verbose=0)[0][0]
        self.result = 'Dog' if prediction > 0.5 else 'Cat'
        
        return img_array[0]  # Return the processed image for display

# Streamlit UI setup
st.title("Real-Time Dog vs Cat Detection")
st.write("Activate your webcam and predict if the captured image is of a dog or a cat.")

# Initialize the WebRTC streamer with the video processor class
ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

# Display the prediction result in Streamlit
if ctx.video_processor:
    st.write(f"Prediction: {ctx.video_processor.result}")
