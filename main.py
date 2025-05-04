from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import depthai as dai
import cv2
import numpy as np
import json
import math
import datetime
import time
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Initialize Firebase
cred = credentials.Certificate(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load model config
with open("best.json", "r") as f:
    model_config = json.load(f)

LABELS = model_config["mappings"]["labels"]
INPUT_SIZE = tuple(map(int, model_config["nn_config"]["input_size"].split("x")))
CONF_THRESHOLD = model_config["nn_config"]["NN_specific_metadata"]["confidence_threshold"]

# Camera and logging config
HFOV = math.radians(81)
VFOV = math.radians(63)
Z = 2.0  # Estimated distance
CAMERA_1_ID = "184430107132EDF400"
CAMERA_2_ID = "184430104131F3F400"
LOG_INTERVAL_SECONDS = 2

# Global state
latest_metrics = {
    "1": {"x": 0.0, "y": 0.0, "confidence": 0.0, "label": "none"},
    "2": {"x": 0.0, "y": 0.0, "confidence": 0.0, "label": "none"},
}
last_log_time = {"1": 0, "2": 0}

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(*INPUT_SIZE)
    cam.setInterleaved(False)
    cam.setFps(30)

    nn = pipeline.createYoloDetectionNetwork()
    nn.setBlobPath("best_openvino_2022.1_6shave.blob")
    nn.setConfidenceThreshold(CONF_THRESHOLD)
    nn.setNumClasses(len(LABELS))
    nn.setCoordinateSize(4)
    nn.setIouThreshold(model_config["nn_config"]["NN_specific_metadata"]["iou_threshold"])

    cam.preview.link(nn.input)

    xout_video = pipeline.createXLinkOut()
    xout_video.setStreamName("video")
    cam.preview.link(xout_video.input)

    xout_nn = pipeline.createXLinkOut()
    xout_nn.setStreamName("detections")
    nn.out.link(xout_nn.input)

    return pipeline

def pixel_to_meter(dx, dy, frame_w, frame_h):
    meters_per_pixel_x = (2 * Z * math.tan(HFOV / 2)) / frame_w
    meters_per_pixel_y = (2 * Z * math.tan(VFOV / 2)) / frame_h
    return dx * meters_per_pixel_x, dy * meters_per_pixel_y

def log_to_firestore(camera_id, label, confidence, dx_m, dy_m):
    detection = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "camera": camera_id,
        "label": label,
        "confidence": confidence,
        "x": dx_m,
        "y": dy_m,
    }
    db.collection("detections").add(detection)

def video_stream(mxid, cam_key):
    pipeline = create_pipeline()
    matching_device = next(
        (dev for dev in dai.Device.getAllAvailableDevices() if dev.getMxId() == mxid),
        None
    )
    if matching_device is None:
        raise RuntimeError(f"Device with MxID {mxid} not found")

    with dai.Device(pipeline, matching_device) as device:
        video = device.getOutputQueue("video", maxSize=4, blocking=False)
        detections = device.getOutputQueue("detections", maxSize=4, blocking=False)

        while True:
            frame = video.get().getCvFrame()
            dets = detections.get().detections

            frame_h, frame_w = frame.shape[:2]
            frame_center = (frame_w // 2, frame_h // 2)
            cv2.circle(frame, frame_center, 5, (0, 0, 255), -1)

            if dets:
                top_det = max(dets, key=lambda d: d.confidence)
                x1 = int(top_det.xmin * frame_w)
                y1 = int(top_det.ymin * frame_h)
                x2 = int(top_det.xmax * frame_w)
                y2 = int(top_det.ymax * frame_h)
                box_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                dx_px = box_center[0] - frame_center[0]
                dy_px = frame_center[1] - box_center[1]
                dx_m, dy_m = pixel_to_meter(dx_px, dy_px, frame_w, frame_h)

                label = LABELS[top_det.label] if top_det.label < len(LABELS) else str(top_det.label)
                confidence = float(top_det.confidence)

                latest_metrics[cam_key] = {
                    "x": dx_m,
                    "y": dy_m,
                    "confidence": confidence,
                    "label": label
                }

                # ✅ Throttled logging to Firestore
                now = time.time()
                if now - last_log_time[cam_key] >= LOG_INTERVAL_SECONDS:
                    log_to_firestore(cam_key, label, confidence, dx_m, dy_m)
                    last_log_time[cam_key] = now

                # Draw overlay
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame, box_center, 5, (255, 0, 0), -1)
                cv2.line(frame, frame_center, box_center, (255, 255, 0), 2)
                cv2.putText(frame, f"X: {dx_m:.2f} m | Y: {dy_m:.2f} m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            _, jpeg = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

# API endpoints
@app.get("/video_feed1")
def video_feed1():
    return StreamingResponse(video_stream(CAMERA_1_ID, "1"), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed2")
def video_feed2():
    return StreamingResponse(video_stream(CAMERA_2_ID, "2"), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/metrics1")
def get_metrics1():
    return latest_metrics["1"]

@app.get("/metrics2")
def get_metrics2():
    return latest_metrics["2"]
