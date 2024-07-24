import os
import cv2
import shutil
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

model = YOLO("/home/vertex/Documents/vertex_projects/ultralytics/ultralytics/models/yolo/detect/yolov8n.pt")
names = model.names
im0 = cv2.imread('/home/vertex/Documents/vertex_projects/ultralytics/examples/temp/Screenshot from 2023-12-21 16-09-08.png')
crop_dir_name = "img_crop"
if not os.path.exists(crop_dir_name):
    os.mkdir(crop_dir_name)
results = model.predict(source=im0)
boxes = results[0].boxes.xyxy.cpu().tolist()
clss = results[0].boxes.cls.cpu().tolist()
confidences = results[0].boxes.conf.cpu().tolist()
combined = list(zip(boxes, confidences))
top_3 = combined[:3]
print(top_3)
folder_path = 'runs'
shutil.rmtree(folder_path,ignore_errors=True)
cv2.waitKey(0)
