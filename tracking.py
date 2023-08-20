import numpy as np
import torch
import yaml
from utils import data, boxes


class Tracking:
    def __init__(self, model, tracker, polygons, args):
        self.model = model
        self.tracker = tracker
        self.polygons = polygons
        self.setting = args

    def load_model(self, weights_path, classes, device='cpu'):
        with torch.no_grad():
            self.device = select_device(device)
            self.model = attempt_load(weights_path, device=self.device)

            if device != "cpu":
                self.model.half()
                self.model.to(self.device).eval()

            stride = int(self.model.stride.max())
            self.img_size = check_img_size()
            self.classes = yaml.load(open(classes), Loader=yaml.SafeLoader)["classes"]

    def unload(self):
        if self.device.type != "cpu":
            torch.cuda.empty_cache()

    def set(selfm, **config):
        pass

    def _preprocess_image(self, img):
        img0 = img.copy()
        img = data.letterbox(img0, self.img_size, auto=self.img_size!=1280)
        img = img[:, :, ::-1]
        if self.setting.rgb_means:
            img -= (0.485, 0.456, 0.406),
        if self.setting.rgb_std:
            img /= (0.229, 0.224, 0.225)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.device.type!="cpu" else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

    def _postprocess_image(self, pred):
        return boxes.postprocess(prediction=pred,
                                 conf_thre=self.setting.conf_thre,
                                 nms_thre=self.setting.nms_thre,
                                 multi_label=False,
                                 agnostic=False,
                                 )

    def detect(self, img, track=False):
        with torch.no_grad():
            org_img, img = self._preprocess_image(img)
            pred = self.model(img)
            pred = self._postprocess_image(pred)
            raw_detection = np.empty((0, 6), float)

            for det in pred:
                if len(det) > 0:
                    det[:, :4] = boxes.scale_coords(img.shape[2:], det[:, :4], org_img.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        raw_detection = np.concatenate((raw_detection, [
                            [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), round(float(conf), 2), int(cls)]]))

            raw_detection = self.tracker.update(raw_detection)

