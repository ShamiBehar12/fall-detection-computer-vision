# fall-detection-computer-vision
Computer vision MVP for human fall detection using MediaPipe keypoints and temporal buffering
# Human Fall Detection – Computer Vision MVP

Computer Vision MVP designed to detect human falls using pose estimation and temporal analysis.

## Overview
This project implements a fall detection algorithm based on human pose keypoints extracted with MediaPipe. A custom temporal buffer was designed to capture motion patterns over time and accurately detect fall events.

The model was trained and evaluated using a dataset of over 2,000 images, achieving 96% classification accuracy.

## Methodology
- Human pose estimation using MediaPipe keypoints
- Feature extraction from body joints and angles
- Custom temporal buffering algorithm to model fall dynamics
- Supervised classification for fall vs non-fall detection

## Tech Stack
- Python
- MediaPipe
- OpenCV
- NumPy
- Scikit-learn
-FASTAPI
## Results
- Accuracy: 96%
- Dataset size: 2,000+ labeled images
- Robust detection of fall events across different body positions

## Notes
This project was developed as an MVP focused on algorithm design and temporal modeling rather than production deployment.
