import numpy as np
import cv2
import math
import yaml


# padding image following stride
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return np.ceil(x / divisor) * divisor


def check_img_size(img_size: np.ndarray, stride=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(stride))  # ceil gs-multiple
    if (new_size != img_size).any():
        print('WARNING: --img-size %s must be multiple of max stride %g, updating to %s' % (str(img_size), stride, str(new_size)))
    return new_size.astype(np.int32).tolist()


def class2id(classes_path):
    classes = {}
    for idx, class_ in enumerate(yaml.load(open(classes_path), Loader=yaml.SafeLoader)["classes"]):
        classes[class_["name"]] = idx
    return classes