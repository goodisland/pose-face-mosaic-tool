# Privacy-friendly CV visualization (Pose skeleton + Face mosaic)

Generate a publishable gif from an image sequence:
- Pose estimation: skeleton-only overlay
- Face detection: mosaic anonymization
- Offline 3-stage pipeline with JSONL cache (fast re-rendering)
- Optional tiled inference + NMS merge
  
## Demo

![demo](output_demo/demo.gif)

## Requirements
- Python 3.8+
- ultralytics
- opencv-python
- numpy
- (Optional) NVIDIA GPU + CUDA-enabled PyTorch for faster inference

## Install
```bash
pip install ultralytics opencv-python numpy
```

### GPU (recommended)

This script can run on CPU, but a CUDA-enabled GPU is strongly recommended for speed.
Install PyTorch with CUDA following the official instructions:
https://pytorch.org/get-started/locally/

## Model weights (.pt)
This repo does **not** include model weights.

Place these files locally (paths are configurable at the top of the script):
- `weights/yolo26x-pose.pt`
- `weights/yolov12l-face.pt`

### Download links
- YOLO26x Pose: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt
- YOLOv12l Face: https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12l-face.pt

Notes:
- You can use any Ultralytics-compatible **pose** and **face** models (the filenames above are examples).
- Please follow each model’s license/terms when downloading and using weights.


## Usage
Prepare an image sequence in a directory (e.g., frames like `000001.jpg`, `000002.jpg`, ...).

1) Set `INPUT_DIR` to the folder that contains the images  
2) (Optional) Set `POSE_MODEL_PATH` / `FACE_MODEL_PATH`  
3) Run:
```bash
python make_pose_face_mosaic_video.py
```

## Notes
- Model weights are NOT included in this repo.
- If you use Ultralytics models, please follow their license/terms.