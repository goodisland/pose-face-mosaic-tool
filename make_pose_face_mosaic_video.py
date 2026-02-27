#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dir_pose_to_video_with_pose_face_tiled_tmp.py

Reads images in a directory in order and:
A) Runs pose inference and saves results to tmp (supports tiled inference)
B) Runs face detection and saves results to tmp (supports tiled inference)
C) Loads tmp results and renders the final output (skeleton overlay + face mosaic)
   then writes an output video (mp4).

Key points
- Pose and Face are split into separate stages (inference vs rendering)
- Intermediate results are cached as JSONL in tmp
- For both Pose and Face: run full-frame + tiled inference, then merge with NMS
- Pose rendering draws skeleton only (no person bounding boxes)
- Faces are anonymized with pixelated mosaic
- Pose/Face have their own independent thresholds
"""

import os
import re
import glob
import sys
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# =========================================================
# Settings (edit only here)
# =========================================================
INPUT_DIR = r".\MOT20\test\MOT20-04\img1"
OUTPUT_VIDEO = "./output/pose_face_mosaic.mp4"

# Models
POSE_MODEL_PATH = r"weights\yolo26x-pose.pt"
FACE_MODEL_PATH = r"weights\yolov12l-face.pt"

FPS = 15.0
DEVICE = None                  # None / "cpu" / "0"
FOURCC = "mp4v"                # if it fails, try "XVID" + .avi
LINE_WIDTH = 2

# -------------------------
# Pose settings
# -------------------------
IMGSZ_POSE = 1280
CONF_POSE_DET = 0.08           # detection confidence threshold for pose inference
IOU_POSE_DET = 0.45            # NMS IoU in Ultralytics for pose inference
POSE_DRAW_KPT_THR = 0.05       # keypoint confidence threshold for drawing
POSE_DRAW_PERSON_THR = 0.00    # person acceptance threshold (bbox conf); 0 disables

ENABLE_TILED_POSE_DET = True
POSE_TILE_ROWS = 4
POSE_TILE_COLS = 4
POSE_TILE_OVERLAP_RATIO = 0.20
POSE_MERGE_NMS_IOU = 0.50      # NMS IoU for merging full-frame + tiled persons (bbox-based)
POSE_MIN_BOX_AREA = 64         # filter out too-small person boxes (px^2)

# -------------------------
# Face settings
# -------------------------
IMGSZ_FACE = 640
CONF_FACE_DET = 0.15
IOU_FACE_DET = 0.25
FACE_PAD = 0.10                # padding ratio around face box

ENABLE_TILED_FACE_DET = True
FACE_TILE_ROWS = 4
FACE_TILE_COLS = 4
FACE_TILE_OVERLAP_RATIO = 0.20
FACE_MERGE_NMS_IOU = 0.50

# Face mosaic
ENABLE_FACE_MOSAIC = True
MOSAIC_BLOCK = 10              # mosaic block size (px). Larger => more pixelated

# tmp cache (intermediate results)
TMP_DIR = "./temp_pose_face_cache"
POSE_TMP_JSONL = os.path.join(TMP_DIR, "pose_results.jsonl")
FACE_TMP_JSONL = os.path.join(TMP_DIR, "face_results.jsonl")

# Save rendered frames (optional)
SAVE_FRAMES = False
FRAMES_DIR = "./output/annotated_frames"

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
# =========================================================

# COCO keypoint skeleton (0-indexed)
COCO_KPT_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]


# =========================================================
# Common utilities
# =========================================================
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def collect_images(input_dir):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    return sorted(set(files), key=natural_key)


def ensure_parent_dir(filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))


def write_jsonl(path, rows):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl_to_dict(path, key_field="index"):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row[key_field]] = row
    return out


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms_boxes_xyxy(boxes_with_conf, iou_thr=0.5):
    """
    boxes_with_conf: [(x1,y1,x2,y2,conf), ...]
    """
    if not boxes_with_conf:
        return []

    boxes = sorted(boxes_with_conf, key=lambda x: x[4], reverse=True)
    kept = []

    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [b for b in boxes if iou_xyxy(best[:4], b[:4]) < iou_thr]

    return kept


def generate_tiles(img_w, img_h, rows=2, cols=2, overlap_ratio=0.2):
    tiles = []

    base_tw = max(1, img_w // cols)
    base_th = max(1, img_h // rows)
    ov_w = int(base_tw * overlap_ratio)
    ov_h = int(base_th * overlap_ratio)

    for r in range(rows):
        for c in range(cols):
            x1 = c * base_tw - ov_w
            y1 = r * base_th - ov_h
            x2 = img_w if c == cols - 1 else (c + 1) * base_tw + ov_w
            y2 = img_h if r == rows - 1 else (r + 1) * base_th + ov_h

            x1 = int(clamp(x1, 0, img_w - 1))
            y1 = int(clamp(y1, 0, img_h - 1))
            x2 = int(clamp(x2, 1, img_w))
            y2 = int(clamp(y2, 1, img_h))

            if x2 > x1 and y2 > y1:
                tiles.append((x1, y1, x2, y2))

    return tiles


# =========================================================
# Face (detection + mosaic)
# =========================================================
def apply_mosaic(image, x1, y1, x2, y2, block=16):
    """
    Apply pixelated mosaic to the specified region.
    """
    h, w = image.shape[:2]
    x1 = int(clamp(x1, 0, w))
    x2 = int(clamp(x2, 0, w))
    y1 = int(clamp(y1, 0, h))
    y2 = int(clamp(y2, 0, h))

    if x2 <= x1 or y2 <= y1:
        return image

    roi = image[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh < 2 or rw < 2:
        return image

    block = max(2, int(block))
    small_w = max(1, int(round(rw / float(block))))
    small_h = max(1, int(round(rh / float(block))))

    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
    mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = mosaic
    return image


def get_face_boxes_from_detection(face_result, img_w, img_h, pad_ratio=0.08, offset_x=0, offset_y=0):
    out = []
    if face_result is None or face_result.boxes is None or face_result.boxes.xyxy is None:
        return out

    xyxy = face_result.boxes.xyxy
    conf = face_result.boxes.conf if face_result.boxes.conf is not None else None

    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu().numpy()
    else:
        xyxy = np.asarray(xyxy)

    if conf is not None:
        conf = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)

    for i, b in enumerate(xyxy):
        x1, y1, x2, y2 = [float(v) for v in b[:4]]
        c = float(conf[i]) if conf is not None and i < len(conf) else 1.0

        x1 += offset_x
        x2 += offset_x
        y1 += offset_y
        y2 += offset_y

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        px = bw * pad_ratio
        py = bh * pad_ratio

        x1 -= px
        x2 += px
        y1 -= py
        y2 += py

        x1 = int(clamp(round(x1), 0, img_w - 1))
        y1 = int(clamp(round(y1), 0, img_h - 1))
        x2 = int(clamp(round(x2), 1, img_w))
        y2 = int(clamp(round(y2), 1, img_h))

        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2, c))
    return out


def detect_faces_tiled(face_model, img_bgr, img_w, img_h):
    all_boxes = []

    # Full-frame inference
    full_results = face_model.predict(
        source=img_bgr,
        conf=CONF_FACE_DET,
        iou=IOU_FACE_DET,
        imgsz=IMGSZ_FACE,
        device=DEVICE,
        verbose=False
    )
    all_boxes.extend(get_face_boxes_from_detection(full_results[0], img_w, img_h, FACE_PAD, 0, 0))

    # Tiled inference
    if ENABLE_TILED_FACE_DET:
        tiles = generate_tiles(img_w, img_h, FACE_TILE_ROWS, FACE_TILE_COLS, FACE_TILE_OVERLAP_RATIO)
        for (tx1, ty1, tx2, ty2) in tiles:
            tile = img_bgr[ty1:ty2, tx1:tx2]
            if tile.size == 0:
                continue

            t_results = face_model.predict(
                source=tile,
                conf=CONF_FACE_DET,
                iou=IOU_FACE_DET,
                imgsz=IMGSZ_FACE,
                device=DEVICE,
                verbose=False
            )
            all_boxes.extend(
                get_face_boxes_from_detection(t_results[0], img_w, img_h, FACE_PAD, tx1, ty1)
            )

    return nms_boxes_xyxy(all_boxes, FACE_MERGE_NMS_IOU)


# =========================================================
# Pose (extract / merge / render)
# =========================================================
def _safe_to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _pose_bbox_from_keypoints(person_kpts, kpt_thr=0.01):
    """
    person_kpts: [[x,y,conf], ...]
    Returns: (x1,y1,x2,y2) or None
    """
    pts = []
    for kp in person_kpts:
        if len(kp) < 3:
            continue
        x, y, c = kp[0], kp[1], kp[2]
        if c >= kpt_thr:
            pts.append((x, y))
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


def extract_pose_persons_from_result(pr, offset_x=0, offset_y=0):
    """
    Convert an Ultralytics pose result into a list of persons.
    Each person:
      {"kpts":[[x,y,c],...], "bbox":[x1,y1,x2,y2], "score":float}
    """
    persons = []

    if pr is None or pr.keypoints is None or pr.keypoints.data is None:
        return persons

    kpts_arr = _safe_to_numpy(pr.keypoints.data)  # (N, K, 3)
    if kpts_arr is None or kpts_arr.ndim != 3:
        return persons

    # Use boxes.conf as a person score if available
    box_conf = None
    if hasattr(pr, "boxes") and pr.boxes is not None and getattr(pr.boxes, "conf", None) is not None:
        box_conf = _safe_to_numpy(pr.boxes.conf)

    for idx, person in enumerate(kpts_arr):
        kpts = []
        for kp in person:
            x = float(kp[0]) + float(offset_x)
            y = float(kp[1]) + float(offset_y)
            c = float(kp[2]) if len(kp) >= 3 else 1.0
            kpts.append([x, y, c])

        bbox = _pose_bbox_from_keypoints(kpts, kpt_thr=0.01)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < POSE_MIN_BOX_AREA:
            continue

        score = float(box_conf[idx]) if box_conf is not None and idx < len(box_conf) else 1.0
        persons.append({
            "kpts": kpts,
            "bbox": [x1, y1, x2, y2],
            "score": score
        })

    return persons


def merge_pose_persons_nms(persons, iou_thr=0.5):
    """
    Merge pose persons using bbox-based NMS (remove duplicates).
    persons: [{"kpts":..., "bbox":[...], "score":...}, ...]
    """
    if not persons:
        return []

    sorted_idx = sorted(range(len(persons)), key=lambda i: persons[i].get("score", 0.0), reverse=True)
    keep = []

    for idx in sorted_idx:
        p = persons[idx]
        pb = p["bbox"]
        score = p.get("score", 0.0)
        if POSE_DRAW_PERSON_THR > 0 and score < POSE_DRAW_PERSON_THR:
            continue

        duplicated = False
        for kp in keep:
            if iou_xyxy(pb, kp["bbox"]) >= iou_thr:
                duplicated = True
                break
        if not duplicated:
            keep.append(p)

    return keep


def detect_pose_tiled(pose_model, img_bgr, img_w, img_h):
    """
    Run pose inference on full-frame + tiles, then merge persons.
    Returns: [{"kpts":[[x,y,c],...], "bbox":[x1,y1,x2,y2], "score":...}, ...]
    """
    all_persons = []

    # Full-frame inference
    full_results = pose_model.predict(
        source=img_bgr,
        conf=CONF_POSE_DET,
        iou=IOU_POSE_DET,
        imgsz=IMGSZ_POSE,
        device=DEVICE,
        verbose=False
    )
    all_persons.extend(extract_pose_persons_from_result(full_results[0], 0, 0))

    # Tiled inference
    if ENABLE_TILED_POSE_DET:
        tiles = generate_tiles(img_w, img_h, POSE_TILE_ROWS, POSE_TILE_COLS, POSE_TILE_OVERLAP_RATIO)
        for (tx1, ty1, tx2, ty2) in tiles:
            tile = img_bgr[ty1:ty2, tx1:tx2]
            if tile.size == 0:
                continue

            t_results = pose_model.predict(
                source=tile,
                conf=CONF_POSE_DET,
                iou=IOU_POSE_DET,
                imgsz=IMGSZ_POSE,
                device=DEVICE,
                verbose=False
            )
            all_persons.extend(extract_pose_persons_from_result(t_results[0], tx1, ty1))

    merged = merge_pose_persons_nms(all_persons, POSE_MERGE_NMS_IOU)

    # Clamp coordinates into the image bounds
    for p in merged:
        for kp in p["kpts"]:
            kp[0] = float(clamp(kp[0], 0, img_w - 1))
            kp[1] = float(clamp(kp[1], 0, img_h - 1))
        x1, y1, x2, y2 = p["bbox"]
        p["bbox"] = [
            float(clamp(x1, 0, img_w - 1)),
            float(clamp(y1, 0, img_h - 1)),
            float(clamp(x2, 1, img_w)),
            float(clamp(y2, 1, img_h))
        ]

    return merged


def serialize_pose_persons(persons):
    """
    Serialize pose persons for JSON saving.
    """
    out = {"persons": []}
    for p in persons:
        out["persons"].append({
            "kpts": [[float(v) for v in kp[:3]] for kp in p.get("kpts", [])],
            "bbox": [float(v) for v in p.get("bbox", [0, 0, 0, 0])],
            "score": float(p.get("score", 0.0))
        })
    return out


def draw_pose_only(frame_bgr, pose_cache, line_width=2, kpt_thr=0.05):
    """
    Draw skeleton only from cached pose results (no bounding boxes).
    """
    if not pose_cache or "persons" not in pose_cache:
        return frame_bgr

    # Draw lines first
    for person in pose_cache["persons"]:
        kpts = person.get("kpts", [])
        n = len(kpts)
        for a, b in COCO_KPT_CONNECTIONS:
            if a >= n or b >= n:
                continue
            xa, ya, ca = kpts[a]
            xb, yb, cb = kpts[b]
            if ca < kpt_thr or cb < kpt_thr:
                continue
            pt1 = (int(round(xa)), int(round(ya)))
            pt2 = (int(round(xb)), int(round(yb)))
            cv2.line(frame_bgr, pt1, pt2, (0, 255, 0), line_width, cv2.LINE_AA)

    # Draw keypoints
    radius = max(2, line_width + 1)
    for person in pose_cache["persons"]:
        kpts = person.get("kpts", [])
        for kp in kpts:
            if len(kp) < 3:
                continue
            x, y, c = kp[0], kp[1], kp[2]
            if c < kpt_thr:
                continue
            cv2.circle(frame_bgr, (int(round(x)), int(round(y))), radius, (0, 0, 255), -1, cv2.LINE_AA)

    return frame_bgr


# =========================================================
# Stage 1: Pose -> tmp
# =========================================================
def stage1_run_pose_and_save(image_files, width, height):
    print("[STAGE 1/3] Pose inference (full + tiles) -> tmp cache")
    pose_model = YOLO(POSE_MODEL_PATH)
    rows = []

    for i, img_path in enumerate(image_files, 1):
        img = cv2.imread(img_path)
        if img is None:
            rows.append({"index": i, "file": img_path, "ok": False, "pose": {"persons": []}})
            continue

        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

        try:
            persons = detect_pose_tiled(pose_model, img, width, height)
            pose_dict = serialize_pose_persons(persons)
            rows.append({"index": i, "file": img_path, "ok": True, "pose": pose_dict})
        except Exception as e:
            rows.append({"index": i, "file": img_path, "ok": False, "error": str(e), "pose": {"persons": []}})

        if i % 10 == 0 or i == len(image_files):
            print("[INFO] Pose tmp progress: {}/{}".format(i, len(image_files)))

    write_jsonl(POSE_TMP_JSONL, rows)
    print("[INFO] Saved pose tmp: {}".format(POSE_TMP_JSONL))


# =========================================================
# Stage 2: Face -> tmp
# =========================================================
def stage2_run_face_and_save(image_files, width, height):
    print("[STAGE 2/3] Face detection (full + tiles) -> tmp cache")

    try:
        face_model = YOLO(FACE_MODEL_PATH)
    except Exception as e:
        print("[ERROR] Failed to load FACE_MODEL_PATH")
        print("       {}".format(FACE_MODEL_PATH))
        print("       {}".format(e))
        sys.exit(1)

    rows = []

    for i, img_path in enumerate(image_files, 1):
        img = cv2.imread(img_path)
        if img is None:
            rows.append({"index": i, "file": img_path, "ok": False, "faces": []})
            continue

        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

        try:
            boxes = detect_faces_tiled(face_model, img, width, height)
            faces = []
            for b in boxes:
                faces.append({
                    "x1": int(b[0]), "y1": int(b[1]),
                    "x2": int(b[2]), "y2": int(b[3]),
                    "conf": float(b[4])
                })
            rows.append({"index": i, "file": img_path, "ok": True, "faces": faces})
        except Exception as e:
            rows.append({"index": i, "file": img_path, "ok": False, "error": str(e), "faces": []})

        if i % 10 == 0 or i == len(image_files):
            print("[INFO] Face tmp progress: {}/{}".format(i, len(image_files)))

    write_jsonl(FACE_TMP_JSONL, rows)
    print("[INFO] Saved face tmp: {}".format(FACE_TMP_JSONL))


# =========================================================
# Stage 3: tmp -> render
# =========================================================
def stage3_render_from_tmp(image_files, width, height):
    print("[STAGE 3/3] Load tmp -> render and write video")

    ensure_parent_dir(OUTPUT_VIDEO)
    if SAVE_FRAMES:
        os.makedirs(FRAMES_DIR, exist_ok=True)

    pose_map = read_jsonl_to_dict(POSE_TMP_JSONL, "index")
    face_map = read_jsonl_to_dict(FACE_TMP_JSONL, "index")

    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (width, height))
    if not writer.isOpened():
        print("[ERROR] Failed to open VideoWriter")
        print("        OUTPUT_VIDEO={}, FOURCC={}".format(OUTPUT_VIDEO, FOURCC))
        print("        Try FOURCC='XVID' and OUTPUT_VIDEO='*.avi'")
        sys.exit(1)

    success_count = 0
    skip_count = 0
    total = len(image_files)

    for i, img_path in enumerate(image_files, 1):
        img = cv2.imread(img_path)
        if img is None:
            print("[WARN] Skip unreadable image: {}".format(img_path))
            skip_count += 1
            continue

        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

        try:
            frame_out = img.copy()

            # Pose rendering (no boxes)
            pose_row = pose_map.get(i, {})
            frame_out = draw_pose_only(
                frame_out,
                pose_row.get("pose", {"persons": []}),
                line_width=LINE_WIDTH,
                kpt_thr=POSE_DRAW_KPT_THR
            )

            # Face mosaic (no boxes)
            if ENABLE_FACE_MOSAIC:
                face_row = face_map.get(i, {})
                for fb in face_row.get("faces", []):
                    frame_out = apply_mosaic(
                        frame_out,
                        fb["x1"], fb["y1"], fb["x2"], fb["y2"],
                        MOSAIC_BLOCK
                    )

            writer.write(frame_out)
            success_count += 1

            if SAVE_FRAMES:
                out_name = "{:06d}_{}.jpg".format(i, Path(img_path).stem)
                cv2.imwrite(os.path.join(FRAMES_DIR, out_name), frame_out)

        except Exception as e:
            print("[WARN] Render failed on image: {}".format(img_path))
            print("       {}".format(e))
            skip_count += 1

        if i % 10 == 0 or i == total:
            print("[INFO] Render progress: {}/{}".format(i, total))

    writer.release()
    print("[INFO] Done.")
    print("[INFO] Saved video: {}".format(OUTPUT_VIDEO))
    print("[INFO] Success: {}, Skipped: {}".format(success_count, skip_count))


# =========================================================
# Main
# =========================================================
def main():
    if not os.path.isdir(INPUT_DIR):
        print("[ERROR] INPUT_DIR not found: {}".format(INPUT_DIR))
        sys.exit(1)

    image_files = collect_images(INPUT_DIR)
    if not image_files:
        print("[ERROR] No images found in: {}".format(INPUT_DIR))
        sys.exit(1)

    os.makedirs(TMP_DIR, exist_ok=True)

    first = cv2.imread(image_files[0])
    if first is None:
        print("[ERROR] Failed to read first image: {}".format(image_files[0]))
        sys.exit(1)
    height, width = first.shape[:2]

    print("[INFO] Found {} images".format(len(image_files)))
    print("[INFO] Frame size: {}x{}".format(width, height))
    print("[INFO] tmp dir: {}".format(TMP_DIR))

    # 1) Pose inference -> tmp
    stage1_run_pose_and_save(image_files, width, height)

    # 2) Face detection -> tmp
    stage2_run_face_and_save(image_files, width, height)

    # 3) Load tmp -> render -> write video
    stage3_render_from_tmp(image_files, width, height)


if __name__ == "__main__":
    main()