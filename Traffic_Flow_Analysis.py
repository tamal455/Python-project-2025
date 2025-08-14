import cv2
import time
import os
from ultralytics import YOLO
import csv
import datetime
import matplotlib.pyplot as plt
import numpy as np

try:
    # Load YOLOv8 model (COCO pretrained)
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Fix the video path - use forward slashes or double backslashes
video_path = r"D:\Python project@2025\myvenv\4K Road traffic video for object detection and tracking - free download now! (1).mp4"

# Check if video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    print("Please check the file path and ensure the video exists.")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    print("Try using a different video file or check video codec.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video info: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")

# Ensure fps is valid
if fps <= 0:
    fps = 30  # Default fallback
frame_delay = int(1000 / fps)

# Create output directory
output_dir = "output_video"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Create plots directory
plots_dir = "plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created plots directory: {plots_dir}")

# Generate timestamp for file naming
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# Set up video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = os.path.join(output_dir, f"vehicle_counts_{timestamp}.mp4")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

if not out.isOpened():
    print("Error: Could not open video writer.")
    print("Trying alternative codec...")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_path = os.path.join(output_dir, f"vehicle_counts_{timestamp}.avi")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"Output video will be saved as: {output_video_path}")

# Lane vertical boundaries (split into 3)
lane1_x = width // 3
lane2_x = (width // 3) * 2

# Horizontal counting line position (adjust as needed)
line_y = int(height * 0.7)

lane_counts = [0, 0, 0]  # Lane 1, Lane 2, Lane 3
counted_ids = set()
object_positions = {}

# Store detailed counting information for CSV export
counted_vehicles = []
counted_lane = {}
counted_frame = {}

# NEW: Data for matplotlib visualizations
vehicle_types = {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0}
hourly_counts = []  # For time-based analysis
lane_timeline = [[], [], []]  # Track counts over time for each lane

frame_count = 0

print("Starting video processing... Press 'q' to quit")

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame")
        break

    frame_count += 1
    
    try:
        # Use YOLO tracking mode (ByteTrack) for unique IDs
        results = model.track(frame, persist=True, verbose=False)

        # Check if any detections exist
        if results and len(results) > 0 and results[0].boxes is not None:
            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                for obj_id, (x1, y1, x2, y2), cls_id, conf in zip(ids, boxes, classes, confs):
                    label = model.names[int(cls_id)]
                    
                    # Filter for vehicles with higher confidence
                    if label in ['car', 'bus', 'truck', 'motorcycle'] and conf > 0.5:
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        # Draw detection + ID
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} ID:{int(obj_id)}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

                        # Improved counting logic
                        if obj_id not in counted_ids:
                            # Store previous position
                            prev_y = object_positions.get(obj_id, cy)
                            object_positions[obj_id] = cy
                            
                            # Count when crossing line from top to bottom
                            if prev_y < line_y and cy >= line_y:
                                # Determine which lane
                                if cx < lane1_x:
                                    lane_number = 1
                                    lane_counts[0] += 1
                                elif cx < lane2_x:
                                    lane_number = 2
                                    lane_counts[1] += 1
                                else:
                                    lane_number = 3
                                    lane_counts[2] += 1
                                
                                # Store counting details for CSV
                                counted_ids.add(obj_id)
                                counted_lane[obj_id] = lane_number
                                counted_frame[obj_id] = frame_count
                                timestamp_sec = frame_count / fps
                                
                                # Update vehicle type counts for matplotlib
                                vehicle_types[label] += 1
                                
                                # Store timeline data
                                lane_timeline[lane_number - 1].append(timestamp_sec)
                                
                                # Store detailed info
                                counted_vehicles.append({
                                    'vehicle_id': int(obj_id),
                                    'vehicle_type': label,
                                    'lane_number': lane_number,
                                    'frame_count': frame_count,
                                    'timestamp': round(timestamp_sec, 2),
                                    'confidence': round(float(conf), 2),
                                    'center_x': cx,
                                    'center_y': cy
                                })
                                
                                print(f"Counted vehicle ID {int(obj_id)} ({label}) in lane {lane_number} at frame {frame_count}")

    except Exception as e:
        print(f"Error processing frame {frame_count}: {e}")
        continue

    # Draw horizontal counting line
    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
    cv2.putText(frame, "COUNTING LINE", (width//2 - 100, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw vertical lane dividers
    cv2.line(frame, (lane1_x, 0), (lane1_x, height), (255, 255, 0), 2)
    cv2.line(frame, (lane2_x, 0), (lane2_x, height), (255, 255, 0), 2)

    # Display lane labels
    cv2.putText(frame, "LANE 1", (lane1_x//2 - 40, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "LANE 2", (lane1_x + (lane2_x-lane1_x)//2 - 40, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "LANE 3", (lane2_x + (width-lane2_x)//2 - 40, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Display counts with better formatting
    cv2.rectangle(frame, (10, 40), (250, 180), (0, 0, 0), -1)
    cv2.putText(frame, f"Lane 1: {lane_counts[0]}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Lane 2: {lane_counts[1]}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Lane 3: {lane_counts[2]}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display frame info
    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (width - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Write frame to output video
    out.write(frame)

    # Show video (optional)
    cv2.imshow("YOLOv8 Multi-Lane Vehicle Counter", frame)

    # Frame rate control
    elapsed_time = (time.time() - start_time) * 1000
    wait_time = max(1, frame_delay - int(elapsed_time))
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

# Final results
print("\n=== FINAL RESULTS ===")
print(f"Lane 1: {lane_counts[0]} vehicles")
print(f"Lane 2: {lane_counts[1]} vehicles") 
print(f"Lane 3: {lane_counts[2]} vehicles")
print(f"Total: {sum(lane_counts)} vehicles")

# Export to CSV
csv_filename = f"vehicle_counts_{timestamp}.csv"
with open(csv_filename, mode='w', newline='') as csvfile:
    fieldnames = ['vehicle_id', 'vehicle_type', 'lane_number', 'frame_count', 'timestamp', 'confidence', 'center_x', 'center_y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for vehicle_data in counted_vehicles:
        writer.writerow(vehicle_data)

print(f"Vehicle count data exported to {csv_filename}")

# Create summary CSV
summary_csv = f"lane_summary_{timestamp}.csv"
with open(summary_csv, mode='w', newline='') as csvfile:
    fieldnames = ['lane_number', 'vehicle_count', 'percentage']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    total = sum(lane_counts)
    for i, count in enumerate(lane_counts, 1):
        percentage = (count / total * 100) if total > 0 else 0
        writer.writerow({
            'lane_number': i,
            'vehicle_count': count,
            'percentage': round(percentage, 2)
        })

print(f"Lane summary exported to {summary_csv}")

# NEW: Create matplotlib visualizations
print("Generating matplotlib visualizations...")

# 1. Lane Distribution Pie Chart
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
lane_labels = ['Lane 1', 'Lane 2', 'Lane 3']
colors = ['#ff9999', '#66b3ff', '#99ff99']
plt.pie(lane_counts, labels=lane_labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Vehicle Distribution by Lane')

# 2. Vehicle Type Bar Chart
plt.subplot(2, 2, 2)
types = list(vehicle_types.keys())
counts = list(vehicle_types.values())
plt.bar(types, counts, color=['blue', 'green', 'orange', 'red'])
plt.title('Vehicle Types Detected')
plt.xlabel('Vehicle Type')
plt.ylabel('Count')
plt.xticks(rotation=45)

# 3. Lane Counts Bar Chart
plt.subplot(2, 2, 3)
plt.bar(lane_labels, lane_counts, color=colors)
plt.title('Vehicles Counted per Lane')
plt.xlabel('Lane')
plt.ylabel('Vehicle Count')

# 4. Traffic Flow Timeline
plt.subplot(2, 2, 4)
video_duration = total_frames / fps
time_bins = np.linspace(0, video_duration, 20)  # 20 time intervals
hist_data = []
for i, timeline in enumerate(lane_timeline):
    if timeline:  # Only if there's data
        hist, _ = np.histogram(timeline, bins=time_bins)
        hist_data.append(hist)
    else:
        hist_data.append(np.zeros(len(time_bins)-1))

time_centers = (time_bins[:-1] + time_bins[1:]) / 2
for i, hist in enumerate(hist_data):
    plt.plot(time_centers, hist, label=f'Lane {i+1}', marker='o')

plt.title('Traffic Flow Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Vehicles/Interval')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f'traffic_analysis_{timestamp}.png'), dpi=300, bbox_inches='tight')
plt.show()

# Additional detailed plot
plt.figure(figsize=(12, 8))

# Vehicle timeline scatter plot
plt.subplot(2, 1, 1)
for vehicle in counted_vehicles:
    color = {'car': 'blue', 'bus': 'green', 'truck': 'orange', 'motorcycle': 'red'}[vehicle['vehicle_type']]
    plt.scatter(vehicle['timestamp'], vehicle['lane_number'], 
               c=color, s=50, alpha=0.7, label=vehicle['vehicle_type'])

# Remove duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title('Vehicle Detection Timeline')
plt.xlabel('Time (seconds)')
plt.ylabel('Lane Number')
plt.yticks([1, 2, 3])
plt.grid(True, alpha=0.3)

# Cumulative count over time
plt.subplot(2, 1, 2)
cumulative_counts = [0, 0, 0]
time_points = []
lane1_cumulative = []
lane2_cumulative = []
lane3_cumulative = []

for vehicle in sorted(counted_vehicles, key=lambda x: x['timestamp']):
    cumulative_counts[vehicle['lane_number'] - 1] += 1
    time_points.append(vehicle['timestamp'])
    lane1_cumulative.append(cumulative_counts[0])
    lane2_cumulative.append(cumulative_counts[1])
    lane3_cumulative.append(cumulative_counts[2])

if time_points:  # Only plot if we have data
    plt.plot(time_points, lane1_cumulative, label='Lane 1', color='red', linewidth=2)
    plt.plot(time_points, lane2_cumulative, label='Lane 2', color='blue', linewidth=2)
    plt.plot(time_points, lane3_cumulative, label='Lane 3', color='green', linewidth=2)

plt.title('Cumulative Vehicle Count Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Cumulative Count')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f'detailed_timeline_{timestamp}.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"Plots saved in {plots_dir}/ directory")

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print("\n=== Processing Complete ===")
print(f"Files created:")
print(f"  - Video: {output_video_path}")
print(f"  - Detailed CSV: {csv_filename}")
print(f"  - Summary CSV: {summary_csv}")
print(f"  - Analysis plots: {plots_dir}/traffic_analysis_{timestamp}.png")
print(f"  - Timeline plots: {plots_dir}/detailed_timeline_{timestamp}.png")
