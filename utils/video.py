from dataclasses import dataclass
from typing import Callable, Generator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    """
    A class to store video information, including width, height, fps and
        total number of frames.

    Attributes:
        width (int): width of the video in pixels
        height (int): height of the video in pixels
        fps (int): frames per second of the video
        total_frames (int, optional): total number of frames in the video,
            default is None

    Examples:
        ```python
        >>> import supervision as sv

        >>> video_info = sv.VideoInfo.from_video_path(video_path='video.mp4')

        >>> video_info
        VideoInfo(width=3840, height=2160, fps=25, total_frames=538)

        >>> video_info.resolution_wh
        (3840, 2160)
        ```
    """

    width: int
    height: int
    fps: int
    total_frames: Optional[int] = None

    @classmethod
    def from_video(cls, video):
        assert video.isOpened(), f"Could not open video."

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        return VideoInfo(width, height, fps, total_frames)

    @property
    def resolution_wh(self):
        return np.array([self.width, self.height])


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)\

    dim = (width, height)

    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)