from ultralytics.models.yolo.segment.predict import SegmentationPredictor


if __name__ =="__main__":
    args = dict(model='yolov8n-seg.pt', source='/home/sangam/Pictures/1.jpg')
    predictor = SegmentationPredictor(overrides=args)
    predictor.predict_cli()
    predictor.show()