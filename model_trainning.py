from ultralytics import YOLO
import multiprocessing


def main():
    model = YOLO("yolo11s.pt")
    model.train(data="coco8.yaml", epochs=100)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()