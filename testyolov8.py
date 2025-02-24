import cv2
import time
import streamlit as st
import numpy as np

# Cấu hình giao diện
st.title("📹 Ứng dụng Theo Dõi Chuyển Động")
st.sidebar.header("Cấu hình")

# Các thanh trượt điều chỉnh tham số
min_contour_area = st.sidebar.slider("Diện tích viền tổng quát", 50, 1000, 200)
threshold_value = st.sidebar.slider("Độ nhạy (Threshold)", 5, 50, 15)
scale_factor = 0.7
blur_size = (11, 11)

# Nút điều khiển
start_button = st.sidebar.button("🚀 Bắt đầu theo dõi")
stop_button = st.sidebar.button("⛔ Dừng")

# Trạng thái chương trình
st.sidebar.text("Trạng thái:")
status = st.sidebar.empty()

# Biến kiểm soát luồng video
running = False

# Hàm xử lý phát hiện chuyển động
def motion_detection():
    global running
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Không thể mở video!")
        return

    running = True
    status.text("🔍 Đang theo dõi chuyển động...")

    # Đọc frame đầu tiên
    ret, prev_frame = cap.read()
    if not ret:
        st.error("❌ Không thể đọc khung hình!")
        return
    prev_frame = cv2.resize(prev_frame, None, fx=scale_factor, fy=scale_factor)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, blur_size, 0)

    start_time = time.time()
    frame_count = 0

    video_container = st.empty()  # Vùng hiển thị video

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        orig_frame = frame.copy()
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, blur_size, 0)

        # So sánh sự khác biệt giữa hai khung hình
        delta_frame = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(delta_frame, threshold_value, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)

        # Tìm các đường viền (contours)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False

        for contour in contours:
            if cv2.contourArea(contour) < min_contour_area:
                continue
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            scaled_x, scaled_y = int(x / scale_factor), int(y / scale_factor)
            scaled_w, scaled_h = int(w / scale_factor), int(h / scale_factor)
            cv2.rectangle(orig_frame, (scaled_x, scaled_y), 
                          (scaled_x + scaled_w, scaled_y + scaled_h), (0, 255, 0), 2)

        # FPS calculation
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Hiển thị FPS và trạng thái chuyển động
        cv2.putText(orig_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(orig_frame, f"Motion: {'YES' if motion_detected else 'NO'}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 0, 255) if motion_detected else (0, 255, 0), 2)

        # Hiển thị hình ảnh lên Streamlit
        orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
        video_container.image(orig_frame, channels="RGB", use_container_width=True)

        prev_gray = gray.copy()

    cap.release()
    status.text("⏹ Đã dừng theo dõi.")
    running = False

# Xử lý khi nhấn nút
if start_button:
    motion_detection()
elif stop_button:
    running = False
    status.text("⏹ Đã dừng theo dõi.")

