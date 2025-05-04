import numpy as np
import tensorrt as trt
import cv2

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load TensorRT engine
def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("best.engine")
context = engine.create_execution_context()

# Bindings
input_idx = engine.get_binding_index(engine.get_binding_name(0))
output_idx = engine.get_binding_index(engine.get_binding_name(1))

input_shape = tuple(engine.get_binding_shape(input_idx))  # (1, 3, 640, 640)
output_shape = tuple(engine.get_binding_shape(output_idx))  # (1, 5, 8400)

# Prepare dummy input (simulated image)
dummy_image = np.random.rand(1, 3, 640, 640).astype(np.float32)

# Allocate output
output_tensor = np.empty(output_shape, dtype=np.float32)

# Run inference
context.set_binding_shape(0, dummy_image.shape)
bindings = [dummy_image.ctypes.data, output_tensor.ctypes.data]
context.execute_v2(bindings)

# Inspect output
print("Output shape:", output_tensor.shape)
print("Output sample:", output_tensor.flatten()[:10])
print("Max:", np.max(output_tensor), "| Min:", np.min(output_tensor))
