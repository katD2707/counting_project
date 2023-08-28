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
from tracker.byte_tracker.core import ByteTracker
from tracker.center_tracker.centroid_tracker import CentroidTracker
from tracker.center_tracker.trackableobject import TrackableObject
from utils.video import VideoInfo

###### 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
###### Token for loading
POLYGONS =  [
    np.array([
        [540,  985 ],
        [1620, 985 ],
        [2160, 1920],
        [1620, 2855],
        [540,  2855],
        [0,    1920]
    ], np.int32),
    np.array([
        [0,    1920],
        [540,  985 ],
        [0,    0   ]
    ], np.int32),
    np.array([
        [1620, 985 ],
        [2160, 1920],
        [2160,    0]
    ], np.int32),
    np.array([
        [540,  985 ],
        [0,    0   ],
        [2160, 0   ],
        [1620, 985 ]
    ], np.int32),
    np.array([
        [0,    1920],
        [0,    3840],
        [540,  2855]
    ], np.int32),
    np.array([
        [2160, 1920],
        [1620, 2855],
        [2160, 3840]
    ], np.int32),
    np.array([
        [1620, 2855],
        [540,  2855],
        [0,    3840],
        [2160, 3840]
    ], np.int32)
]

def parse_arguments():
    # function to parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=False,
                    help="path to config file")
    ap.add_argument("-f", "--exp_file", type=str, default="yolox/version/mot/yolox_tiny_mix_det.py")
    ap.add_argument("-ckpt", type=str, default="weights/bytetrack_tiny_mot17.pth.tar")
    ap.add_argument("-i", "--input", type=str, default="video_test/market-square.mp4",
                    help="path to optional input video file")
    ap.add_argument("-o", "--output_path", type=str, default="results/square_ver2.mp4",
                    help=" path to optional output video file")
    ap.add_argument("-n", "--name", type=str, default=None,
                    help="Experiment name")
    ap.add_argument("-cn", "--class_name", type=str, default="person")
    ap.add_argument("-cp", "--classes_path", type=str, default="classes/coco.yaml")
    ap.add_argument("-d", "--device", type=str, default="cuda:0")

    # Model parameter
    ap.add_argument("--model_size_hw", type=list, default=960,
                    help="Size of image fitting to the model, the smaller, the quicker")
    ap.add_argument("--rgb_means", type=bool, default=True,
                    help="Standardize input before fit to model")
    ap.add_argument("--rgb_std", type=bool, default=True,
                    help="Standardize input before fit to model")

    # Tracker parameter
    ap.add_argument("--track_thres", type=float, default=0.25)
    ap.add_argument("--track_buffer", type=int, default=30)
    ap.add_argument("--match_thres", type=float, default=0.8)

    # Detection parameter
    ap.add_argument("--conf_thres", type=float, default=0.5)
    # ap.add_argument("--track_buffer", type=int, default=30)
    ap.add_argument("--nms_thres", type=float, default=0.5)

    args = ap.parse_args()

    return args


def main(logger):
    # main function for people_counter.py
    args = parse_arguments()
    exp = get_exp(args.exp_file, args.name)

    # if a video path was not supplied, give a reference to the ip camera
    if not args.input:
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
    tracker = ByteTracker(args, frame_rate=30)

    # Get video information
    video_info = VideoInfo.from_video(capture)

    # # Initialize tracking
    track = Tracking(logger=logger, model=model, video_info=video_info, tracker=tracker, args=args)
    track.load_model(weights_path=args.ckpt, classes_path=args.classes_path, device=args.device)
    track.init_drawer(polygons=POLYGONS)

    vid_writer = cv2.VideoWriter(
        args.output_path, cv2.VideoWriter_fourcc(*"mp4v"), video_info.fps, (int(video_info.width), int(video_info.height))
    )

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
        # print(outputs[0].shape)
        # # If no object is detected
        online_targets = torch.Tensor([[-1, -1, -1, -1, -1, -1, -1]])
        if outputs[0] is not None:
            online_targets = track.track(outputs[0], image_infos, class_name=args.class_name)

        # Annotate video with tracked objects
        frame = track.annotate(frame, online_targets)
        vid_writer.write(frame)
        # # show the output frame
        # cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
        # key = cv2.waitKey(1) & 0xFF
        #
        # # if the `q` key was pressed, break from the loop
        # if key == ord("q"):
        #     break

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
