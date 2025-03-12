# LEGO Detection with YOLOv8

## Overview
This project is a Gradio web app that uses a trained YOLOv8 model to detect LEGO pieces in uploaded images.


## Demo
Gradio link


## Guidance
1. Run the following command to create virtual environment:

`pip install -r requirements.txt`

2. Run `dataset_process.py` to pre-process the data, including reduce dataset, convert to single label, convert to YOLO format label file, and split dataset into train, val and test.

3. Run `model_training.py` to train the model based on `yolov8n.pt`.

The trained model will be stored as `runs/detect/train/weights/best.pt`.

4. Run `model_performance.py` to calculate mean average precision (mAP) when the IoU threshold is set to 0.5.


## Dataset
Full dataset: https://www.kaggle.com/datasets/dreamfactor/biggest-lego-dataset-600-parts

The model used in this project is trained with 1,000 images randomly seleted from this dataset. Modify `image_num` in `dataset_process.py` to define a different dataset size.


## Model Performance
mAP@0.5: 0.9891