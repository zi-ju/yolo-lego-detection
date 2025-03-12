from ultralytics import YOLO

# Load the best trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Validate on the dataset and compute metrics
metrics = model.val(data="lego.yaml", iou=0.5)

# Get mAP@0.5
mAP_50 = metrics.box.map50
print(f"mAP@0.5: {mAP_50:.4f}")
