# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops
import cv2

classes = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train",
           7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant", 11: "stop sign", 12: "parking meter",
           13: "bench", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant",
           21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie",
           28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
           34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
           39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana",
           47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
           54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
           61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
           68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
           75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush", 80: "number_plate",
           81: "number_plate"}



class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """
    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
            # annotated_frame = results[0].plot()
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()
            combined = list(zip(boxes, clss, confidences))
            for box, cls, conf in combined:
                class_name = classes[int(cls)]
                confidence = conf
                xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
                cv2.rectangle(orig_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 165, 255), 2)
                cv2.putText(orig_img, f'{class_name} {confidence:.2f}', (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # Optionally, show only the top 3 detections
            top_3 = combined[:3]
            # print(top_3)
            cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
            cv2.imshow("YOLOv8 Inference", orig_img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        return results


if __name__ == "__main__":
    # args = dict(model='yolov8n.pt', source='rtsp://admin:nepal@123@192.168.1.64')
    args = dict(model='/home/vertex/Downloads/merged.pt', source='/home/vertex/Downloads/costliest-number-plates.jpg')
    #args = dict(model='/home/vertex/Downloads/best.pt', source='0')
    # args = dict(model='yolov8n.pt', source='/home/sangam/Downloads/MicrosoftTeams-video.mp4')
    # args = dict(model='yolov8n.pt', source='/home/sangam/Downloads/input.jpg')6
    predictor = DetectionPredictor(overrides=args)
    # predictor = DetectionPredictor(model='/home/vertex/Downloads/with_background.pt',source='/home/vertex/Downloads/images (15).jpeg')
    #predictor = DetectionPredictor(model='yolov8n.pt',source='/home/vertex/Documents/vertex_projects/ultralytics/examples/temp/Screenshot from 2023-12-21 16-09-08.png')
    predictor.predict_cli()
    # predictor.show()
