# Privacy-friendly CV visualization (Pose skeleton + Face mosaic)

Generate a publishable MP4 from an image sequence:
- Pose estimation: skeleton-only overlay
- Face detection: mosaic anonymization
- Offline 3-stage pipeline with JSONL cache (fast re-rendering)
- Optional tiled inference + NMS merge

## Requirements
- Python 3.8+
- ultralytics
- opencv-python
- numpy

## Install
```bash
pip install ultralytics opencv-python numpy
```

## Usage
Edit the settings at the top of the script (INPUT_DIR, model paths, etc.) and run:
```bash
python make_pose_face_mosaic_video.py
```

## Notes
- Model weights are NOT included in this repo.
- If you use Ultralytics models, please follow their license/terms.