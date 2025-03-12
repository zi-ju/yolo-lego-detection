import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/detect/train/weights/best.pt")


def detect_lego(image):
    # Convert PIL image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Run inference
    results = model(image_cv)

    # Draw bounding boxes on the image
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()  # Confidence score
            label = "LEGO"

            # Draw rectangle
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_cv, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert back to RGB format for display
    output_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    return output_image


# Create Gradio interface
demo = gr.Interface(
    fn=detect_lego,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="LEGO Detector",
    description="Upload an image, and the trained YOLOv8 model will detect LEGO pieces."
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
