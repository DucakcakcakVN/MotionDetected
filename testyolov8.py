import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np

class MotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.prev_gray = None
        self.min_contour_area = 200
        self.threshold_value = 15
        self.blur_size = (11, 11)
        self.scale_factor = 0.7

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.prev_gray is None:
            resized = cv2.resize(img, None, fx=self.scale_factor, fy=self.scale_factor)
            self.prev_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            self.prev_gray = cv2.GaussianBlur(self.prev_gray, self.blur_size, 0)
            return img

        resized = cv2.resize(img, None, fx=self.scale_factor, fy=self.scale_factor)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_size, 0)

        delta_frame = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(delta_frame, self.threshold_value, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            scaled_x, scaled_y = int(x / self.scale_factor), int(y / self.scale_factor)
            scaled_w, scaled_h = int(w / self.scale_factor), int(h / self.scale_factor)
            cv2.rectangle(img, (scaled_x, scaled_y),
                          (scaled_x + scaled_w, scaled_y + scaled_h), (0, 255, 0), 2)

        self.prev_gray = gray
        return img

def main():
    st.title("Ứng dụng theo dõi chuyển động")
    st.write("Made by Grok-3")
    webrtc_streamer(key="example", video_transformer_factory=MotionTransformer)

if __name__ == "__main__":
    main()
