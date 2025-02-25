import cv2 
import streamlit as st
import numpy as np
import tempfile

cap = cv2.VideoCapture(0)
st.title("Ứng dụng Theo Dõi Chuyển Động")
frame_placeholder = st.empty()
stop_button_pressed = st.empty()
while cap.isOpened() and not stop_button_pressed:
    ret, frame = cap.read()
    if not ret:
        st.write("Không thể truy cập webcam.")
        break
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, CHANNELS="RGB")

    if cv2.waitKey(1) & 0xFF == ord('q') or stop_button_pressed:
        break
cap.release()
