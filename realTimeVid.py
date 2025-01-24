import cv2
import torch
import pyttsx3
import os
from datetime import datetime
import time

# Initialize the Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speech speed

# Load YOLOv5 model (pretrained)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5s

# Function for text-to-speech alerts
def speak_alert(alert):
    print(alert)  # Print alert to the console
    engine.say(alert)  # Speak the alert
    engine.runAndWait()

# Initialize tracking for alerts (to avoid repeated messages)
tracked_alerts = {}
time_threshold = 10  # Minimum time (in seconds) between repeated alerts for the same object

# Define alert rules for specific objects
def handle_detection(label, bbox, image, confidence):
    x_min, y_min, x_max, y_max = bbox
    color = (0, 255, 0)  # Default color: green
    message = None

    # Unique ID for the object based on its bounding box and label
    object_id = f"{label}_{x_min}_{y_min}_{x_max}_{y_max}"

    if label == 'person':
        # Check if the person is a pedestrian (not on a vehicle)
        height = y_max - y_min
        width = x_max - x_min
        aspect_ratio = height / width  # Pedestrians tend to have a high aspect ratio

        if aspect_ratio > 1.5:  # Likely a pedestrian (standing upright)
            color = (0, 0, 255)  # Red for pedestrians
            message = "Alert: Pedestrian detected! Slow down."
        else:
            return  # Skip people on vehicles or those with low aspect ratios

    elif label == 'traffic light':
        # Identify traffic light color based on its region of interest
        traffic_light_roi = image[y_min:y_max, x_min:x_max]
        hsv = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2HSV)

        # Define HSV ranges for red, yellow, and green
        red_lower = (0, 120, 70)
        red_upper = (10, 255, 255)
        green_lower = (40, 40, 40)
        green_upper = (70, 255, 255)
        yellow_lower = (20, 100, 100)
        yellow_upper = (30, 255, 255)

        # Check for colors in the traffic light area
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        if red_mask.sum() > green_mask.sum() and red_mask.sum() > yellow_mask.sum():
            color = (0, 0, 255)  # Red for stop
            message = "Alert: Red light! Stop."
        elif green_mask.sum() > red_mask.sum() and green_mask.sum() > yellow_mask.sum():
            color = (0, 255, 0)  # Green for go
            message = "Alert: Green light! Proceed."
        elif yellow_mask.sum() > red_mask.sum() and yellow_mask.sum() > green_mask.sum():
            color = (0, 255, 255)  # Yellow for caution
            message = "Caution: Yellow light! Slow down."

    elif label == 'pothole':
        color = (255, 0, 0)  # Blue for potholes
        message = "Alert: Pothole detected! Drive carefully."

    # Draw bounding box and label
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(image, f"{label} ({confidence:.2f})", (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Trigger alert if needed, and only if it hasn't been triggered recently
    current_time = time.time()
    if message:
        last_alert_time = tracked_alerts.get(object_id, 0)
        if current_time - last_alert_time > time_threshold:
            speak_alert(message)
            tracked_alerts[object_id] = current_time

# Load the input video
video_path = "E:\AMJ PROJECTS\AI-Traffic-Analysis\ds3.mp4"  # Replace with the path to your input video
cap = cv2.VideoCapture(video_path)

# Specify the output folder
output_folder = "E:\AMJ PROJECTS\AI-Traffic-Analysis\output\Vid"  # Replace with your folder path
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

# Generate a unique filename based on the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"road_analysis_{timestamp}.avi"

# Full output path
output_path = os.path.join(output_folder, output_file)

# Define video writer with the generated file path
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Process each frame of the video
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Bounding boxes, confidence, class, name

    for _, row in detections.iterrows():
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence, label = row['confidence'], row['name']

        # Filter detections for specific objects with confidence > 0.5
        if confidence > 0.6 and label in ['person', 'traffic light', 'pothole']:
            handle_detection(label, (x_min, y_min, x_max, y_max), frame, confidence)

    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame in a window
    cv2.imshow("Road Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

    frame_id += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to: {output_path}")
