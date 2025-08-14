Traffic Flow Analysis

This project analyzes traffic flow from a given video i.e (4K Road traffic video for object detection and tracking) by detecting and counting vehicles in three distinct lanes using computer vision. It leverages YOLOv8 for object detection and OpenCV for video processing.

Features

Detect vehicles using YOLOv8 (pre-trained on COCO dataset).

Count vehicles separately for Lane 1, Lane 2, and Lane 3.

Track vehicles across frames to avoid double-counting.

Overlay lane boundaries and live vehicle counts on the video.


Requirements

Make sure you have the following installed:

Python 3.8+

pip (Python package manager)

Git (optional, for cloning the repo)

===============================================Steps:-=================================
Create a virtual environment (recommended)
python -m venv venv

Install dependencies
pip install -r requirements.txt

Required Packages
Package	Purpose
ultralytics	Main YOLOv8 framework for vehicle detection
opencv-python	Video processing and lane overlay drawing
torch	Deep learning backend for YOLOv8
numpy	Array and numerical operations
pandas	(Optional) For storing/exporting traffic data
matplotlib	(Optional) Plotting traffic analysis results

Install Command :- :- :- :- :- :-
pip install ultralytics opencv-python torch numpy pandas matplotlib
Example requirements.txt:
opencv-python
ultralytics
torch
numpy

Run the script
Traffic_Flow_Analysis.py

=======================================OUTPUT===========================================


Real-time visual display of:

Lane boundaries

Vehicle detection boxes

Vehicle counts per lane


-: -: -: -: -: -: -: -: -: -: -: -: -: -: -: -: PATH :- :- :- :- :- :- :- :- :- :- :- :- :- :- :- :- :- :- :- :- :-

A processed video saved in output_video folder
AND CSV File Exported will be saved in-side the project folder file name (vehicle_counts.csv & lane_summary.csv)

Sample Video inside Python project@2025\myvenv
file name 4K Road traffic video for object detection and tracking - free download now!(1).mp4
put this file inside the folder called myvenv . then run the Traffic_Flow_Analysis.py script


