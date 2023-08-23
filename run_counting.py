import numpy as np
import argparse
import threading
import schedule
import logging
import time
import dlib
import json
import csv
import cv2
import sys
import os
import torch
import datetime

from tracking import Tracking
from yolox.exp import get_exp
from tracker.byte_tracker.byte_tracker import BYTETracker
from tracker.center_tracker.centroid_tracker import CentroidTracker
from tracker.center_tracker.trackableobject import TrackableObject
from utils.video import VideoInfo


POLYGONS = []

def parse_arguments():
    # function to parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True,
                    help="path to config file")
    ap.add_argument("-f", "--exp_file", type=str, default="yolox/version/mot/yolox_tiny_mix_det.py")
    ap.add_argument("-ckpt", type=str, default="weights/bytetrack_tiny_mot17.pth")
    ap.add_argument("-i", "--input", type=str, default=0,
                    help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
                    help=" path to optional output video file")
    ap.add_argument("-n", "--name", type=str, default=None,
                    help=" path to optional output video file")
    ap.add_argument("-n", "--name", type=str, default="yolox/version/mot/yolox_tiny_mix_det.py")
    ap.add_argument("-cp", "--classes_path", type=str, default="classes\coco.yaml")
    ap.add_argument("-d", "--device", type=str, default="cuda:0")

    args = ap.parse_args()

    return args


def main(logger):
    # main function for people_counter.py
    args = parse_arguments()
    exp = get_exp(args.exp_file, args.name)

    # if a video path was not supplied, give a reference to the ip camera
    if not args.get("input", False):
        logger.info("Starting the live stream...")

        capture = cv2.VideoCapture(args.camera_url)
        # IP camera log in with username and password
        # capture = cv2.VideoCapture('rtsp://username:password@192.168.1.64/1')

        # vs = VideoStream(args.camera_url).start()
        time.sleep(2.0)
    # otherwise, grab a reference to the video file
    else:
        if args.input.isdigit():
            logger.info("Starting the webcam...")
            capture = cv2.VideoCapture(int(args.input))
        else:
            logger.info("Starting the video...")
            if not os.path.isfile(args.input):
                print("Input video file ", args.input, " doesn't exist")
                sys.exit(1)
            capture = cv2.VideoCapture(args.input)

    # Initialize model
    model = exp.get_model()

    # Initialize byte_tracker
    ByteTracker = BYTETracker(args, frame_rate=30)

    # Get video information
    video_info = VideoInfo.from_video(capture)

    # # Initialize tracking
    track = Tracking(logger=logger, model=model, video_info=video_info, tracker=ByteTracker, args=args)
    track.load_model(weights_path=args.ckpt, classes=args.classes_path, device=args.device)
    track.init_drawer(polygons=POLYGONS)

    # loop over frames from the video stream
    while True:
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        ret, frame = capture.read()

        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video
        if args.input is not None and frame is None:
            break

        # Outputs is list of frame, each with detected boxes
        outputs, image_infos = track.detect(img=frame)
        # If no object is detected
        online_targets = torch.Tensor([[-1, -1, -1, -1, -1, -1, -1]])
        if outputs[0] is not None:
            online_targets = track.track(outputs[0], [image_infos['height'], image_infos['width']], class_name=args.class_name)

        # Annotate video with tracked objects
        frame = track.annotate(frame, online_targets)

        # show the output frame
        cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # close any open windows
    cv2.destroyAllWindows()
    track.unload()


if __name__ == "__main__":
    # execution start time
    start_time = time.time()
    # setup logger
    logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
    logger = logging.getLogger(__name__)
    main(logger)
