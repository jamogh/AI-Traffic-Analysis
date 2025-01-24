import cv2
import torch
import os
import pyttsx3

# Initialize the Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech speed

# Load YOLOv5 model (pretrained)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # yolov5s is lightweight and fast

# Function for text-to-speech alerts
def speak_alert(alert):
    engine.say(alert)
    engine.runAndWait()

# Path to the dataset directory
dataset_dir = r"E:\AMJ PROJECTS\AI-Traffic-Analysis\IDDMissingTSMiniTest\IDDMissingTSMiniTest\task1test"  # Replace with your folder path
output_dir = r"E:\AMJ PROJECTS\AI-Traffic-Analysis\output\testmini"  # Directory to save output images
os.makedirs(output_dir, exist_ok=True)

# Verify the dataset directory exists
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"The specified directory does not exist: {dataset_dir}")

# Process each image in the dataset
for image_name in os.listdir(dataset_dir):
    if image_name.endswith('.png'):  # Check for PNG files
        image_path = os.path.join(dataset_dir, image_name)
        image = cv2.imread(image_path)

        # Perform object detection
        results = model(image)
        detections = results.pandas().xyxy[0]  # Bounding boxes, confidence, class, name

        for _, row in detections.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            confidence, class_name = row['confidence'], row['name']

            # Filter detections (e.g., only show objects with >50% confidence)
            if confidence > 0.5:
                # Draw bounding boxes and labels
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(image, f"{class_name} ({confidence:.2f})", (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Trigger alerts for specific objects
                if class_name in ['traffic light', 'person', 'animal', 'pothole']:
                    speak_alert(f"Alert: {class_name} detected in {image_name}!")

        # Save the processed image with bounding boxes
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, image)

        print(f"Processed: {image_name} | Output saved to: {output_path}")

print("Processing completed.")
