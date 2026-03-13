from ultralytics import YOLO

# Load a COCO-pretrained YOLO26m model
model = YOLO("yolo26m.pt")

# Run inference with the YOLO26n model on the 'bus.jpg' image
results = model("./tent.webp")

# Display the results with bounding boxes
results[0].save("output.jpg")