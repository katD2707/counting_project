import numpy as np
import cv2
from typing import Tuple, Optional, List

from geometry import Point, Rect
from color import Color


def polygon_to_mask(polygon: np.ndarray, resolution_wh: Tuple[int, int]) -> np.ndarray:
    """Generate a mask from a polygon.

    Args:
        polygon (np.ndarray): The polygon for which the mask should be generated,
            given as a list of vertices.
        resolution_wh (Tuple[int, int]): The width and height of the desired resolution.

    Returns:
        np.ndarray: The generated 2D mask, where the polygon is marked with
            `1`'s and the rest is filled with `0`'s.
    """
    width, height = resolution_wh
    mask = np.zeros((height, width))
    cv2.fillPoly(mask, [polygon], color=1)
    return mask


def clip_boxes(
    boxes_xyxy: np.ndarray, frame_resolution_wh: Tuple[int, int]
) -> np.ndarray:
    """
    Clips bounding boxes coordinates to fit within the frame resolution.

    Args:
        boxes_xyxy (np.ndarray): A numpy array of shape `(N, 4)` where each
            row corresponds to a bounding box in
        the format `(x_min, y_min, x_max, y_max)`.
        frame_resolution_wh (Tuple[int, int]): A tuple of the form `(width, height)`
            representing the resolution of the frame.

    Returns:
        np.ndarray: A numpy array of shape `(N, 4)` where each row
            corresponds to a bounding box with coordinates clipped to fit
            within the frame resolution.
    """
    result = np.copy(boxes_xyxy)
    width, height = frame_resolution_wh
    result[:, [0, 2]] = result[:, [0, 2]].clip(0, width)
    result[:, [1, 3]] = result[:, [1, 3]].clip(0, height)
    return result


def draw_polygon(
    scene: np.ndarray, polygon: np.ndarray, color: Color, thickness: int = 2
) -> np.ndarray:
    """Draw a polygon on a scene.

    Parameters:
        scene (np.ndarray): The scene to draw the polygon on.
        polygon (np.ndarray): The polygon to be drawn, given as a list of vertices.
        color (Color): The color of the polygon.
        thickness (int, optional): The thickness of the polygon lines, by default 2.

    Returns:
        np.ndarray: The scene with the polygon drawn on it.
    """
    cv2.polylines(
        scene, [polygon], isClosed=True, color=color.as_bgr(), thickness=thickness
    )
    return scene


def draw_text(
    scene: np.ndarray,
    text: str,
    text_anchor: Point,
    text_color: Color = Color.black(),
    text_scale: float = 0.5,
    text_thickness: int = 1,
    text_padding: int = 10,
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    background_color: Optional[Color] = None,
) -> np.ndarray:
    """
    Draw text with background on a scene.

    Parameters:
        scene (np.ndarray): A 2-dimensional numpy ndarray representing an image or scene
        text (str): The text to be drawn.
        text_anchor (Point): The anchor point for the text, represented as a
            Point object with x and y attributes.
        text_color (Color, optional): The color of the text. Defaults to black.
        text_scale (float, optional): The scale of the text. Defaults to 0.5.
        text_thickness (int, optional): The thickness of the text. Defaults to 1.
        text_padding (int, optional): The amount of padding to add around the text
            when drawing a rectangle in the background. Defaults to 10.
        text_font (int, optional): The font to use for the text.
            Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        background_color (Color, optional): The color of the background rectangle,
            if one is to be drawn. Defaults to None.

    Returns:
        np.ndarray: The input scene with the text drawn on it.

    Examples:
        ```python
        >>> scene = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> text_anchor = Point(x=50, y=50)
        >>> scene = draw_text(scene=scene, text="Hello, world!",text_anchor=text_anchor)
        ```
    """
    text_width, text_height = cv2.getTextSize(
        text=text,
        fontFace=text_font,
        fontScale=text_scale,
        thickness=text_thickness,
    )[0]
    text_rect = Rect(
        x=text_anchor.x - text_width // 2,
        y=text_anchor.y - text_height // 2,
        width=text_width,
        height=text_height,
    ).pad(text_padding)

    if background_color is not None:
        scene = draw_filled_rectangle(
            scene=scene, rect=text_rect, color=background_color
        )

    cv2.putText(
        img=scene,
        text=text,
        org=(text_anchor.x - text_width // 2, text_anchor.y + text_height // 2),
        fontFace=text_font,
        fontScale=text_scale,
        color=text_color.as_bgr(),
        thickness=text_thickness,
        lineType=cv2.LINE_AA,
    )
    return scene


def draw_filled_rectangle(scene: np.ndarray, rect: Rect, color: Color) -> np.ndarray:
    """
    Draws a filled rectangle on an image.

    Parameters:
        scene (np.ndarray): The scene on which the rectangle will be drawn
        rect (Rect): The rectangle to be drawn
        color (Color): The color of the rectangle

    Returns:
        np.ndarray: The scene with the rectangle drawn on it
    """
    cv2.rectangle(
        scene,
        rect.top_left.as_xy_int_tuple(),
        rect.bottom_right.as_xy_int_tuple(),
        color.as_bgr(),
        -1,
    )
    return scene




