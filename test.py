import cv2

# Thay link này bằng link IP camera thật của bạn
ip_camera_url = 'rtsp://admin:123456@192.168.100.131:554/h265'  # ví dụ với IP Webcam (Android)
# ip_camera_url = 'rtsp://admin:password@192.168.1.100:554/stream1'  # ví dụ với camera RTSP

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không kết nối được camera IP")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được khung hình")
        break

    cv2.imshow("IP Camera", frame)
    if cv2.waitKey(1) == 27:  # ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()
