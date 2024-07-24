from ultralytics.models.yolo.detect.predict import DetectionPredictor


if __name__ =="__main__":
    # args = dict(model='yolov8n.pt', source='rtsp://admin:nepal@123@192.168.1.64')
    args = dict(model='/home/vertex/Downloads/plant_only.pt', source='/home/vertex/Downloads/dataset-card.jpg')
    predictor = DetectionPredictor(overrides=args)
    predictor.predict_cli()