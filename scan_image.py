import sys
import time
import os
import json
import cv2
import requests
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout,
    QWidget, QMessageBox, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

# ---------- Thread xử lý OCR + YOLO ----------
class OCRWorker(QThread):
    finished = pyqtSignal(str, QImage)

    def __init__(self, frame, model):
        super().__init__()
        self.frame = frame
        self.model = model
        self.ocr = ocr

    def run(self):
        try:
            self.start_time = time.time()
            cv2.imwrite("test.jpg", self.frame)
            results = self.model("test.jpg")
            original_image = cv2.imread("test.jpg")

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped = original_image[y1:y2, x1:x2]
                    cv2.imwrite("license_plate_cropped.jpg", cropped)
                    result = self.ocr.predict(input=cropped)
                    for res in result:
                        res.save_to_json("output/license_plate.json")
                        res.save_to_img("output/license_plate.jpg")

                    with open("output/license_plate.json", 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    rec_texts = ''.join(data['rec_texts'])
                    print(data['rec_texts'])
                    plate = rec_texts if rec_texts else 'Không rõ'

                    url = "http://localhost:8000/api/plate"  # Thay bằng URL thực tế
                    files = {
                        'full_image': open("test.jpg", 'rb'),
                        'plate_image': open("output/license_plate.jpg", 'rb')
                    }
                    data = {
                        'plate_text': plate
                    }
                    response = requests.post(url, files=files, data=data)
                    print(f"📤 Gửi API thành công. Status: {response.status_code}")

                    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                    h, w, ch = rgb.shape
                    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

                    self.finished.emit(plate, qimg)
                    elapsed = time.time() - self.start_time
                    print(f"⏱ Thời gian xử lý: {elapsed:.2f} giây")
                    return

            self.finished.emit("Không phát hiện biển số", QImage())
        except Exception as e:
            self.finished.emit(f"Lỗi: {str(e)}", QImage())

# ---------- Giao diện chính ----------
class LicensePlateApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nhận diện biển số")
        self.setGeometry(100, 100, 640, 500)

        self.model = YOLO("weights/best.pt")

        # Widgets
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Nhập URL camera IP hoặc để trống dùng webcam")

        self.start_button = QPushButton("Bắt đầu camera")
        self.capture_button = QPushButton("Chụp ảnh nhận diện")
        self.image_label = QLabel()
        self.crop_label = QLabel()
        self.result_label = QLabel("Biển số: ")

        self.image_label.setFixedSize(400, 300)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.crop_label.setFixedSize(200, 100)
        self.crop_label.setStyleSheet("border: 1px solid black;")

        layout = QVBoxLayout()
        layout.addWidget(self.url_input)
        layout.addWidget(self.start_button)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.image_label)
        layout.addWidget(QLabel("Ảnh biển số:"))
        layout.addWidget(self.crop_label)
        layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.start_button.clicked.connect(self.toggle_camera)
        self.capture_button.clicked.connect(self.capture_image)

    def toggle_camera(self):
        if self.cap is None:
            url = self.url_input.text().strip()
            if url == "":
                self.cap = cv2.VideoCapture(0)
            else:
                self.cap = cv2.VideoCapture(url)

            if not self.cap.isOpened():
                QMessageBox.critical(self, "Lỗi", "Không mở được camera.")
                self.cap = None
                return

            self.timer.start(30)
            self.start_button.setText("Tắt camera")
        else:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.start_button.setText("Bắt đầu camera")
            self.image_label.clear()

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (400, 300))
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def capture_image(self):
        if not self.cap:
            QMessageBox.warning(self, "Thông báo", "Chưa kết nối camera.")
            return

        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.critical(self, "Lỗi", "Không thể chụp ảnh.")
            return

        self.worker = OCRWorker(frame, self.model)
        self.worker.finished.connect(self.display_result)
        self.worker.start()
        print("Chụp ảnh hoàn tất")

    def display_result(self, plate_text, qimg):
        if not qimg.isNull():
            self.crop_label.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.crop_label.width(), self.crop_label.height(), Qt.KeepAspectRatio))
        self.result_label.setText(f"Biển số: {plate_text}")
        print(f"Biển số: {plate_text}")

# ---------- Chạy ứng dụng ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LicensePlateApp()
    window.show()
    sys.exit(app.exec_())
