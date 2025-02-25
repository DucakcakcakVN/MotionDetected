import cv2
import streamlit as st
import numpy as np
import time

# Hàm phát hiện chuyển động
def detect_motion(frame, prev_gray, min_contour_area, threshold_value, blur_size, scale_factor):
    orig_frame = frame.copy()
    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, blur_size, 0)

    delta_frame = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(delta_frame, threshold_value, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)

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

    return orig_frame, gray, motion_detected

# Ứng dụng Streamlit
def main():
    st.title("Ứng dụng theo dõi chuyển động")
    st.write("Made by Grok-3")

    # Điều khiển giao diện
    min_contour_area = st.sidebar.slider("Diện tích viền tổng quát", 50, 1000, 200)
    threshold_value = st.sidebar.slider("Độ nhạy", 5, 50, 15)
    start_button = st.button("Bắt đầu theo dõi")
    stop_button = st.button("Dừng lại")

    # Placeholder để hiển thị video và trạng thái
    video_placeholder = st.empty()
    status_placeholder = st.empty()

    # Thông số cố định
    blur_size = (11, 11)
    scale_factor = 0.7

    # Trạng thái chạy
    if "running" not in st.session_state:
        st.session_state.running = False

    if start_button:
        st.session_state.running = True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Không thể truy cập webcam.")
            return

        ret, prev_frame = cap.read()
        if not ret:
            st.error("Không thể đọc từ webcam.")
            cap.release()
            return

        prev_frame = cv2.resize(prev_frame, None, fx=scale_factor, fy=scale_factor)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, blur_size, 0)

        start_time = time.time()
        frame_count = 0

        while st.session_state.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Phát hiện chuyển động
            processed_frame, prev_gray, motion_detected = detect_motion(
                frame, prev_gray, min_contour_area, threshold_value, blur_size, scale_factor
            )

            # Tính FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            # Hiển thị FPS và trạng thái trên khung hình
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Motion: {'YES' if motion_detected else 'NO'}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255) if motion_detected else (0, 255, 0), 2)

            # Chuyển đổi sang RGB để hiển thị trên Streamlit
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(processed_frame_rgb, channels="RGB")

            # Cập nhật trạng thái
            status_placeholder.write(f"Trạng thái: {'Hoạt động' if motion_detected else 'Không có chuyển động'}")

            # Dừng nếu nhấn nút "Dừng lại"
            if stop_button:
                st.session_state.running = False
                break

            # Giảm tải CPU bằng cách thêm độ trễ nhỏ
            time.sleep(0.03)  # ~30 FPS tối đa

        cap.release()
        status_placeholder.write("Trạng thái: Đã dừng")
        video_placeholder.empty()

if __name__ == "__main__":
    main()
