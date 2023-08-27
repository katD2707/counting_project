import numpy as np
import torch
import os
import cv2

from utils import boxes, torch_utils
from supervision.detection.core import Detections
import supervision as sv
from utils.data import check_img_size, class2id, letterbox
from supervision import Detections

class Tracking:
    def __init__(self, logger, args, video_info, model=None, tracker=None):
        self.model = model
        self.tracker = tracker
        self.video_info = video_info
        self.setting = args
        self.logger = logger

    def load_model(self, weights_path, classes_path, device='cpu'):
        with torch.no_grad():
            self.device = torch_utils.select_device(device)
            if device != "cpu":
                self.model.half()
                self.model.to(self.device).eval()
                self.logger.info("loading checkpoint")
                ckpt = torch.load(weights_path, map_location=device)
                # load the model state dict
                self.model.load_state_dict(ckpt["model"])
                self.logger.info("loaded checkpoint done.")

            stride = self.model.strides.max()

            self.model_size_hw = check_img_size(np.array(self.setting.model_size_hw), stride=stride)
            self.classes = class2id(classes_path=classes_path)

    def unload(self):
        if self.device.type != "cpu":
            torch.cuda.empty_cache()

    def set(self, **config):
        pass

    def init_drawer(self, polygons):
        colors = sv.ColorPalette.default()
        
        self.zones = [
            sv.PolygonZone(
                polygon=polygon,
                frame_resolution_wh=self.video_info.resolution_wh
            )
            for polygon
            in polygons
        ]
        self.zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone,
                color=colors.by_idx(index),
                thickness=6,
                text_thickness=8,
                text_scale=4
            )
            for index, zone
            in enumerate(self.zones)
        ]
        self.box_annotators = [
            sv.BoxAnnotator(
                color=colors.by_idx(index),
                thickness=4,
                text_thickness=4,
                text_scale=2
            )
            for index
            in range(len(polygons))
        ]

    def _preprocess_image(self, img):
        img = letterbox(img, self.model_size_hw, auto=self.model_size_hw!=1280)
        img = img[:, :, ::-1]

        img = np.divide(img, 255.0, casting="unsafe")
        if self.setting.rgb_means:
            img -= (0.485, 0.456, 0.406),
        if self.setting.rgb_std:
            img /= (0.229, 0.224, 0.225)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.device!="cpu" else img.float()

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img, (img.shape[-2], img.shape[-1])

    def _postprocess_image(self, pred):
        return boxes.postprocess(prediction=pred,
                                 conf_thres=self.setting.conf_thres,
                                 nms_thres=self.setting.nms_thres,
                                 multi_label=False,
                                 agnostic=False,
                                 )

    def detect(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["raw_height"] = height
        img_info["raw_width"] = width
        img_info["raw_img"] = img

        with torch.no_grad():
            # org_img: [h, w, 3]
            # img: [1, 3, h, w]
            img, img_hw = self._preprocess_image(img)
            # Model outputs: [B, n_anchors, detect_attrs]
            # Detections attrs ordered as (x_center, y_center, w, h, obj_conf, class_1, class_2, ...)
            pred = self.model(img)
            # Post-process results ordered as (x1, y1, x2, y2, score, class)
            pred = self._postprocess_image(pred)
        img_info["transform_hw"] = img_hw
        return pred, img_info

    def track(self, preds, img_info, class_name:str=None):
        '''

        :param preds: Tensor [n_anchors, 6]
        :param image_shape: [height, weight]
        :param class_id: provide if only track a specific class. E.g. 0: "person"
        :return: [n_anchors, x1, y1, x2, y2, score, id, class]
        '''
        # Track a specific class
        if class_name:
            preds = preds[preds[:, 5]==self.classes[class_name]]

        img_h, img_w = img_info["height"], img_info["weight"]
        scale = min(img_info["transform_hw"][0] / float(img_h), img_info["transform_hw"][1] / float(img_w))
        preds[:, :4] /= scale
        preds = Detections.from_yolox(yolox_results=preds)

        # Return list of STrack objects
        online_targets = self.tracker.update_with_detections(preds)

        return online_targets

    def annotate(self, frame, detections):
        for zone, zone_annotator, box_annotator in zip(self.zones, self.zone_annotators, self.box_annotators):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
            frame = zone_annotator.annotate(scene=frame)

        return frame





