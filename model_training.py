from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")
    model.train(data="lego.yaml", epochs=50, batch=16, imgsz=640)


if __name__ == "__main__":
    main()
