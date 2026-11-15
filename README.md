# Hotel Facility Detection using YOLO11

## Project Overview
This project detects and counts hotel room facilities using a fine-tuned YOLO11 Nano object detection model.

Users can capture an image using their camera or upload an existing image. The application detects hotel facilities, displays bounding boxes and provides class-wise object counts along with confidence.

## Classes
The model detects 10 classes:

- AC
- Bed
- Chair
- Fan
- Lamp
- Mirror
- Refrigerator
- Sofa
- TV
- Wardrobe

## Technologies Used
- Python
- YOLO11 (Ultralytics)
- OpenCV
- Streamlit
- NumPy
- Pandas
- Matplotlib

## Features
- Capture images using a camera.
- Upload images from a local device.
- Detect hotel facilities using YOLO11.
- Display predicted images with bounding boxes.
- Count objects class-wise.
- Display total detected objects and class-wise summary.
- Visualize object counts using a bar chart.
- Download detection results as a CSV file.

## Project Structure
```text
Hotel_Facility_Detection/
│
├── app.py
├── best.pt
└── requirements.txt
```

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Model
A fine-tuned YOLO11 Nano model (`best.pt`) is used for detecting the 10 hotel facility classes.

## Output
The application displays:
- Original image
- Predicted image with bounding boxes
- Class-wise object counts
- Total objects detected
- Object count visualization
- Downloadable CSV report