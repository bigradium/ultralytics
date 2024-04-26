from ultralytics import YOLO
# Load a model
model = YOLO('../runs/detect/train5/weights/best.pt')  # load an official model
# Export the model
model.export(format='onnx', opset=11)