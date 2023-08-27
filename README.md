# Object Detection and Tracking in Boxing Videos

This project focuses on performing object detection and tracking specifically for boxers within a boxing ring in a video. The goal is to accurately detect boxers, track their movements, and visualize their trajectories using a YOLOv8-based model. The process involves several steps, including data preparation, model training, and trajectory plotting.

## Output
~~[Here]() is the drive link for the results~~ **Redacted due to input video being proprietary**

## Prerequisites

Before starting the project, ensure that you have the following:

- A video file (`input.mp4`) containing the boxing match footage.
- Access to the necessary libraries: `opencv-python`, `numpy`, `ultralytics`, and `ffmpeg`.

## Steps

1. **Data Preparation:**
   - Extract images from the input video using `ffmpeg`.
   - Annotate a subset of images using a tool like Roboflow and augment the dataset using shear and bounding box rotation.

2. **Model Training for Object Detection:**
   - Train the YOLOv8 Nano pretrained model using the annotated dataset.
   - Use the following command to initiate training:
     ```bash
     !yolo task=detect mode=train model=yolov8x.pt data={dataset.location}/data.yaml epochs=25 imgsz=800 plots=True
     ```
   - View the training progress and results through generated plots and images.

3. **Object Tracking and Trajectory Plotting:**
   - Utilize the best-trained YOLOv8 model for object tracking and trajectory plotting.
   - Load the YOLOv8 model using the Ultralytics library.
   - Open the input video and define necessary parameters for the output video.
   - Loop through the video frames, running YOLOv8 tracking on each frame.
   - If detections are found, annotate the frame and track the detected boxers.
   - Plot the trajectories of boxers by connecting their centroid points over time.

4. **Visualization and Output:**
   - The annotated video with trajectories plotted will be saved as `output_video_persistent.mp4`.

## Code Blocks

Below are the code blocks used in this project:

1. YOLOv8 Model Training:
   ```bash
   !yolo task=detect mode=train model=yolov8n.pt data={dataset.location}/data.yaml epochs=25 imgsz=800 plots=True
   ```
2. YOLOv8 Tracking and saving
```python
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('/content/gdrive/MyDrive/Track_Boxing/best8n.pt')

# Open the video file
video_path = "/content/gdrive/MyDrive/Track_Boxing/10.mp4"
cap = cv2.VideoCapture(video_path)

# Get the video's frame width, height, and frames per second (fps)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# Define the codec and create a VideoWriter object
output_path = "/content/gdrive/MyDrive/Track_Boxing/output_video_persistent.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Store the track history
track_history = defaultdict(lambda: [])
track_colors = {}  # Store track IDs and their associated colors

# Define a list of different colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Check if there are any detections in the results
        if results[0].boxes is not None:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = []
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks with different colors
            for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                # if len(track) > 30:  # retain 90 tracks for 90 frames
                #     track.pop(0)

                # Get the track's color or assign a new color
                if track_id not in track_colors:
                    track_colors[track_id] = colors[i % len(colors)]

                color = track_colors[track_id]

                # Draw the tracking lines with the assigned color
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=color, thickness=10)

            # Write the annotated frame to the output video
            out.write(annotated_frame)
        else:
            # Write the original frame to the output video
            out.write(frame)

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and video writer
cap.release()
out.release()

```
