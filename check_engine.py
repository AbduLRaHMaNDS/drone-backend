import cv2
import depthai as dai
import numpy as np
import tensorrt as trt
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import time

app = FastAPI()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

engine = load_engine("drone-detection.engine")
context = engine.create_execution_context()

input_binding_idx = engine.get_binding_index(engine.get_binding_name(0))
output_binding_idx = engine.get_binding_index(engine.get_binding_name(1))

input_shape = tuple(engine.get_binding_shape(input_binding_idx))
output_shape = tuple(engine.get_binding_shape(output_binding_idx))

pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)

device = dai.Device(pipeline)
rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(frame):
    frame_resized = cv2.resize(frame, (640, 640))
    frame_transposed = frame_resized.transpose(2, 0, 1)
    frame_normalized = frame_transposed / 255.0
    input_tensor = np.expand_dims(frame_normalized, axis=0).astype(np.float32)
    return input_tensor

def infer(context, input_tensor):
    input_tensor = np.ascontiguousarray(input_tensor)
    output_tensor = np.empty(output_shape, dtype=np.float32)

    context.set_binding_shape(0, input_tensor.shape)
    bindings = [input_tensor.ctypes.data, output_tensor.ctypes.data]

    context.execute_v2(bindings)
    return output_tensor

def decode_yolo_output(output, frame_shape, conf_threshold=0.5):
    height, width, _ = frame_shape

    output = output.reshape(1, 5, 8400)
    output = np.transpose(output, (0, 2, 1))[0]

    detections = []

    for det in output:
        x_center, y_center, w, h, conf = det

        if np.isnan(det).any():
            continue

        x_center = sigmoid(x_center)
        y_center = sigmoid(y_center)
        conf = sigmoid(conf)

        if conf < conf_threshold:
            continue

        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        detections.append((x1, y1, x2, y2, conf))

    return detections

def generate_frames():
    prev_time = time.time()

    while True:
        in_rgb = rgb_queue.get()
        frame = in_rgb.getCvFrame()

        input_tensor = preprocess(frame)
        output = infer(context, input_tensor)

        detections = decode_yolo_output(output, frame.shape, conf_threshold=0.5)

        for x1, y1, x2, y2, conf in detections:
            label = f"drone {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        cv2.putText(frame, f"x-axis: {center_x}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"y-axis: {center_y}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/camera")
def camera_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
