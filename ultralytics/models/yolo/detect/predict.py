# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops
import cv2


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
            annotated_frame = results[0].plot()
            cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord(''):
                break
        return results


if __name__ == "__main__":
    # args = dict(model='yolov8n.pt', source='rtsp://admin:nepal@123@192.168.1.64')
    args = dict(model='yolov8n.pt', source='0')
    # args = dict(model='yolov8n.pt', source='/home/sangam/Downloads/MicrosoftTeams-video.mp4')
    # args = dict(model='yolov8n.pt', source='/home/sangam/Downloads/input.jpg')
    predictor = DetectionPredictor(overrides=args)
    predictor.predict_cli()
    # predictor.show()
