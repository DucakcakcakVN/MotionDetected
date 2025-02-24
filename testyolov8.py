import cv2
import streamlit as st
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Giao diện Streamlit
st.title("🎥 Ứng dụng Theo Dõi Chuyển Động")
st.sidebar.write("### Cài đặt")
min_contour_area = st.sidebar.slider("Diện tích viền tối thiểu", 50, 1000, 200)
threshold_value = st.sidebar.slider("Độ nhạy phát hiện", 5, 50, 15)

# Bộ xử lý video
class MotionDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.prev_gray = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Phát hiện chuyển động
        delta_frame = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(delta_frame, threshold_value, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_contour_area:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.prev_gray = gray
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC Stream
webrtc_streamer(key="motion-detect", video_processor_factory=MotionDetectionProcessor)
