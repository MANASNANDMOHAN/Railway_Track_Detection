import os
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO

# Function to find the best anchors
def find_best_anchors(data_path, num_anchors=9):
    boxes = []
    for label_file in os.listdir(data_path):
        with open(os.path.join(data_path, label_file)) as f:
            for line in f.readlines():
                _, x, y, w, h = map(float, line.split())
                boxes.append([w, h])
    boxes = np.array(boxes)
    kmeans = KMeans(n_clusters=num_anchors, random_state=0).fit(boxes)
    anchors = kmeans.cluster_centers_
    return anchors

# Define the path to your labels
data_path = 'D:/CODE/Projects/Company_Project1/data/train/labels'
best_anchors = find_best_anchors(data_path)
print("Best anchors found:", best_anchors)

# Load the YOLOv8 model configuration
model = YOLO('yolov8n.yaml')

# Train the model with reduced number of workers
try:
    model.train(data='D:/CODE/Projects/Company_Project1/data/data.yaml', epochs=100, imgsz=640, workers=2)
except RuntimeError as e:
    print(f"RuntimeError during training: {e}")

# Define the path where the best model is saved
best_model_path = 'runs/detect/train/weights/best.pt'

# Check if the best model file exists
if os.path.exists(best_model_path):
    print(f"Best model saved to: {best_model_path}")
else:
    print("Best model path not found")

# Load the best model
model = YOLO(best_model_path)

# Define the NMS threshold (default is usually around 0.45)
nms_threshold = 0.45  # Adjust this value as needed

# Perform predictions with adjusted NMS
results = model.predict(r'D:\CODE\Projects\Company_Project1\data\test\images\frame_209.jpg', conf=0.25, iou=nms_threshold)

# Define fixed bounding box size
fixed_width = 50
fixed_height = 50

# Adjust the bounding boxes to have a fixed size
for result in results:
    for box in result.boxes:
        x_center, y_center, _, _ = box.xywh
        box.xywh = [x_center, y_center, fixed_width, fixed_height]

#Doing Prediction using Trained model On images
from ultralytics import YOLO
import os

# Define paths
best_model_path = 'runs/detect/train/weights/best.pt'
test_images_path = 'D:/CODE/Projects/Company_Project1/data/test/images'  # Adjust this path if necessary
results_save_path = 'runs/detect/test_results/'  # Path to save test results

# Load the best model
model = YOLO(best_model_path)

# Ensure the test results save directory exists
os.makedirs(results_save_path, exist_ok=True)

# Run inference on the test dataset
results = model.predict(source=test_images_path, save=True, project=results_save_path)

# Print summary of results
print(f"Results saved to: {results_save_path}")

#Doing Inferencing on Video
from ultralytics import YOLO
import os

# Define paths
best_model_path = 'runs/detect/train/weights/best.pt'
video_path = 'D:/CODE/Projects/Company_Project1/Videos/R_1.mkv'  # Replace with your video file path
results_save_path = 'runs/detect/video_results/'  # Path to save video results

# Load the best model
model = YOLO(best_model_path)

# Ensure the video results save directory exists
os.makedirs(results_save_path, exist_ok=True)

# Run inference on the video
results = model.predict(source=video_path, save=True, project=results_save_path, name='predicted_video')

# Print summary of results
print(f"Predicted video saved to: {os.path.join(results_save_path, 'predicted_video.mp4')}")
