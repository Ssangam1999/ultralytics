from ultralytics import YOLO

# Load a model
model = YOLO("/home/vertex/Documents/vertex_projects/ultralytics/ultralytics/models/yolo/detect/yolov8n.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(['/home/vertex/Documents/vertex_projects/ultralytics/examples/temp/Screenshot from 2023-12-21 16-09-08.png'])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs