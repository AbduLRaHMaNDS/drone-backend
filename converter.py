from ultralytics import YOLO

# Load your model
model = YOLO("drone-detection.pt")  # Replace "model.pt" with your actual filename if needed

# Export to ONNX
model.export(format="onnx")
