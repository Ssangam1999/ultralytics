from ultralytics.models.yolo.detect.predict import DetectionPredictor


if __name__ =="__main__":
    # args = dict(model='yolov8n.pt', source='rtsp://admin:nepal@123@192.168.1.64')
    args = dict(model='yolov8n.pt', source='/home/sangam/Downloads/maxresdefault.jpg')
    predictor = DetectionPredictor(overrides=args)
    predictor.predict_cli()
    predictor.show()