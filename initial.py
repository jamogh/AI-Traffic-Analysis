import cv2
import numpy as np
import csv
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Pre-trained YOLOv8 nano model for object detection

# Function to detect anomalies
def detect_anomalies(detections, frame_num):
    anomalies = []
    for detection in detections:
        label = detection["label"]
        box = detection["box"]
        
        # Example anomaly: Animal detected on the road
        if label in ["dog", "cat", "horse", "deer"]:
            anomalies.append({
                "frame": frame_num,
                "type": "Animal Crossing",
                "label": label,
                "box": box
            })
        
        # Example anomaly: Pedestrian detected in unauthorized areas
        if label == "person" and (box["x_min"] < 100 or box["x_max"] > 500):  # Example area condition
            anomalies.append({
                "frame": frame_num,
                "type": "Unauthorized Pedestrian",
                "label": label,
                "box": box
            })
        
        # Add more anomaly types as needed
        # Example: Stopped vehicle, abnormal behavior, etc.
    
    return anomalies

# Function to process traffic video and generate anomaly report
def process_traffic_video(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    anomalies_report = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        results = model(frame)  # Perform object detection
        detections = []

        # Parse YOLO results
        for result in results.xyxy[0]:
            xmin, ymin, xmax, ymax, confidence, class_id = result[:6]
            label = model.names[int(class_id)]
            box = {"x_min": int(xmin), "y_min": int(ymin), "x_max": int(xmax), "y_max": int(ymax), "confidence": float(confidence)}
            
            detections.append({"label": label, "box": box})

        # Detect anomalies
        anomalies = detect_anomalies(detections, frame_num)
        anomalies_report.extend(anomalies)

        # Annotate the frame for visualization (optional)
        for detection in detections:
            label = detection["label"]
            box = detection["box"]
            cv2.rectangle(frame, (box["x_min"], box["y_min"]), (box["x_max"], box["y_max"]), (0, 255, 0), 2)
            cv2.putText(frame, label, (box["x_min"], box["y_min"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame (comment this out if running on headless servers)
        cv2.imshow("Traffic Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save anomalies report to CSV
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["frame", "type", "label", "box"])
        writer.writeheader()
        for anomaly in anomalies_report:
            writer.writerow(anomaly)
    
    print(f"Anomalies report saved to {output_csv}")

# Run the program
if __name__ == "__main__":
    video_path = "traffic_video.mp4"  # Replace with your traffic video file path
    output_csv = "anomalies_report.csv"  # Output report file
    process_traffic_video(video_path, output_csv)
