# from ultralytics.models.yolo.detect.predict import DetectionPredictor
import cv2
import os
from ultralytics import YOLO

disease = ['tomato_healthy', 'tomato_yellow_leaf_curl_virus', 'tomato_bacterial_spot', 'potato_healthy', 'potato_early_blight', 'potato_late_blight', 'tomato_septoria_leaf_spot', 'tomato_early_blight', 'tomato_mosaic_virus', 'tomato_late_blight', 'tomato_spider_mites', 'tomato_leaf_mold']

# Path to folder containing images
image_folder = '/home/vertex/Documents/augmented_verified_leaf_dataset/content/dataset/varified_leaf_dataset'

classification_parent_dir = "/home/vertex/Documents/datasets/classification"
object_detection_parent_dir = "/home/vertex/Documents/datasets/obj_detection"


# Path to output folder
# Save crop garera rakhne

#Iterate over each image in the folder
for dir in os.listdir(image_folder):
        if dir==disease[0]:
                directory_path =f"{object_detection_parent_dir}/{disease[0]}"
                if not os.path.isdir(directory_path):
                        os.mkdir(directory_path)
                for img_name in os.listdir(os.path.join(image_folder,dir)):
                        image=os.path.join(image_folder, dir, img_name)
                        args = dict(model='/home/vertex/Downloads/plant_only.pt', source=image)
                        predictor = DetectionPredictor(overrides=args)
                        # print("the predictor is ", predictor)
                        predictor.predict_cli()
                        print("the predictor is ",predictor)

                break
        break


        break

        # for image_file in os.listdir((os.path.join(image_folder, dir))):
        #         img = cv2.imread(os.path.join(image_folder,dir,image_file))
        #         print(img)
        #         break
        # break


                # image_path = os.path.join(image_folder, image_file)
                # args = dict(model='/home/vertex/Downloads/plant_only.pt', source=image_path)
                # predictor = DetectionPredictor(overrides=args)
                # predictor.predict_cli()

