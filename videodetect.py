import cv2
import torch
import numpy as np
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech speed

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Pretrained YOLOv5

# Text-to-speech function
def speak_alert(alert):
    engine.say(alert)
    engine.runAndWait()

# Lane detection function
def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Region of interest for lane detection
    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    roi = np.array([[(50, height), (width - 50, height), (width // 2 + 50, height // 2), (width // 2 - 50, height // 2)]], np.int32)
    cv2.fillPoly(mask, roi, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    # Hough transform to detect lines
    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame

# Process video input
video_path = "video.mp4"  # Replace with your video file
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video processing completed.")
        break

    # Object detection
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Bounding boxes, confidence, class, name

    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence, class_name = row['confidence'], row['name']

        # Filter detections (confidence > 50%)
        if confidence > 0.5:
            # Draw bounding boxes
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({confidence:.2f})", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Trigger alerts for specific objects
            if class_name in ['traffic light', 'person', 'animal', 'pothole']:
                speak_alert(f"Alert: {class_name} detected!")

    # Lane detection
    frame = detect_lanes(frame)

    # Display video with detections
    cv2.imshow("Driving Assistance System", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
