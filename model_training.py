from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")
    model.train(data="lego.yaml", epochs=50, batch=16, imgsz=640)

    results = model.val(data="lego.yaml", imgsz=640, iou_thres=0.5)
    print(results)


if __name__ == "__main__":
    main()
