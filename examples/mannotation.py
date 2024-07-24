from ultralytics.models.yolo.detect.predict import DetectionPredictor


import os


# Path to folder containing images
image_folder = '/home/vertex/Documents/modified_leaf_dataset/Leaf detection.v3i.yolov8/test/another'

# Path to output folder


# Iterate over each image in the folder
for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        args = dict(model='/home/vertex/Downloads/plant_only.pt', source=image_path)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()