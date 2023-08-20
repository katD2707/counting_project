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
# from

# execution start time
start_time = time.time()
# setup logger
logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments():
    # function to parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True,
                    help="path to config file")
    ap.add_argument("-p", )
    ap.add_argument("-m", )
    ap.add_argument("-i", "--input", type=str,
                    help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
                    help=" path to optional output video file")
    args = vars(ap.parse_args())

    return args


def main():
    # main function for people_counter.py
    args = parse_arguments()

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

    # initialize the video writer (we'll instantiate later if need be)
    writer = None

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    width = None
    height = None


