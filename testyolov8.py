import cv2
import streamlit as st
import numpy as np

# 🚀 Giao diện Streamlit
st.title("🎥 Ứng dụng Theo Dõi Chuyển Động với OpenCV")
st.sidebar.write("### Cài đặt")

# Điều chỉnh thông số
min_contour_area = st.sidebar.slider("Diện tích viền tối thiểu", 50, 1000, 200)
threshold_value = st.sidebar.slider("Độ nhạy phát hiện", 5, 50, 15)
blur_size = (11, 11)  # Kích thước bộ lọc làm mờ

# 🔄 Xin quyền truy cập camera trước khi chạy
if "camera_permission" not in st.session_state:
    st.session_state.camera_permission = False

if not st.session_state.camera_permission:
    if st.button("🎥 Bật Camera"):
        st.session_state.camera_permission = True
        st.experimental_rerun()

# 🚦 Nếu đã cấp quyền camera, tiến hành xử lý
if st.session_state.camera_permission:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Không thể truy cập camera. Hãy kiểm tra lại kết nối!")
    else:
        # Bộ nhớ đệm frame trước
        prev_gray = None

        # Chạy vòng lặp chính
        stframe = st.empty()  # Tạo vùng hiển thị video trong Streamlit
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Không thể lấy frame từ camera!")
                break

            # Xử lý hình ảnh
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, blur_size, 0)

            if prev_gray is None:
                prev_gray = gray
                continue

            # So sánh hai frame để phát hiện chuyển động
            delta_frame = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(delta_frame, threshold_value, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Tìm contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) < min_contour_area:
                    continue
                motion_detected = True
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Ghi trạng thái chuyển động
            status_text = "🚨 Phát hiện chuyển động!" if motion_detected else "✅ Không có chuyển động"
            cv2.putText(frame, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Cập nhật frame trước
            prev_gray = gray

            # Hiển thị hình ảnh trên Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Đổi màu BGR -> RGB
            stframe.image(frame, channels="RGB", use_column_width=True)

        # Giải phóng camera sau khi thoát
        cap.release()
        cv2.destroyAllWindows()
