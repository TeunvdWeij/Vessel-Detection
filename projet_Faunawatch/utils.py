import cv2
import os
import numpy as np
import validators
import typing_extensions as tx

import urllib.request
import urllib.parse

import typing
from shapely import geometry
from scipy import spatial

def read(filepath_or_buffer):
    """Read a file into an image object

    Args:
        filepath_or_buffer: The path to the file, a URL, or any object
            with a `read` method (such as `io.BytesIO`)
    """
    if isinstance(filepath_or_buffer, np.ndarray):
        return filepath_or_buffer
    if hasattr(filepath_or_buffer, "read"):
        image = np.asarray(bytearray(filepath_or_buffer.read()), dtype=np.uint8)  # type: ignore
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    elif isinstance(filepath_or_buffer, str):
        if validators.url(filepath_or_buffer):
            return read(urllib.request.urlopen(filepath_or_buffer))
        assert os.path.isfile(filepath_or_buffer), (
            "Could not find image at path: " + filepath_or_buffer
        )
        image = cv2.imread(filepath_or_buffer)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(image, max_scale, max_size):
    """Obtain the optimal resized image subject to a maximum scale
    and maximum size.

    Args:
        image: The input image
        max_scale: The maximum scale to apply
        max_size: The maximum size to return
    """
    if max(image.shape) * max_scale > max_size:
        # We are constrained by the maximum size
        scale = max_size / max(image.shape)
    else:
        # We are contrained by scale
        scale = max_scale
    return (
        cv2.resize(
            image, dsize=(int(image.shape[1] * scale), int(image.shape[0] * scale))
        ),
        scale,
    )

def pad(image, width, height, cval = 255):
    """Pad an image to a desired size. Raises an exception if image
    is larger than desired size.

    Args:
        image: The input image
        width: The output width
        height: The output height
        cval: The value to use for filling the image.
    """
    output_shape: typing.Union[typing.Tuple[int, int, int], typing.Tuple[int, int]]
    if len(image.shape) == 3:
        output_shape = (height, width, image.shape[-1])
    else:
        output_shape = (height, width)
    assert height >= output_shape[0], "Input height must be less than output height."
    assert width >= output_shape[1], "Input width must be less than output width."
    padded = np.zeros(output_shape, dtype=image.dtype) + cval
    padded[: image.shape[0], : image.shape[1]] = image
    return padded


def adjust_boxes(
    boxes,
    scale=1,
    boxes_format: tx.Literal["boxes", "predictions", "lines"] = "boxes",
):
    """Adjust boxes using a given scale and offset.

    Args:
        boxes: The boxes to adjust
        boxes_format: The format for the boxes. See the `drawBoxes` function
            for an explanation on the options.
        scale: The scale to apply
    """
    if scale == 1:
        return boxes
    if boxes_format == "boxes":
        return np.array(boxes) * scale
    if boxes_format == "lines":
        return [
            [(np.array(box) * scale, character) for box, character in line]
            for line in boxes
        ]
    if boxes_format == "predictions":
        return [(word, np.array(box) * scale) for word, box in boxes]
    raise NotImplementedError(f"Unsupported boxes format: {boxes_format}")

def get_rotated_box(points):
    """Obtain the parameters of a rotated box.

    Returns:
        The vertices of the rotated box in top-left,
        top-right, bottom-right, bottom-left order along
        with the angle of rotation about the bottom left corner.
    """
    try:
        mp = geometry.MultiPoint(points=points)
        pts = np.array(list(zip(*mp.minimum_rotated_rectangle.exterior.xy)))[
            :-1
        ]  # noqa: E501
    except AttributeError:
        # There weren't enough points for the minimum rotated rectangle function
        pts = points
    # The code below is taken from
    # https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py

    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = spatial.distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    pts = np.array([tl, tr, br, bl], dtype="float32")

    rotation = np.arctan((tl[0] - bl[0]) / (tl[1] - bl[1]))
    return pts, rotation


def get_rotated_width_height(box):
    """
    Returns the width and height of a rotated rectangle

    Args:
        box: A list of four points starting in the top left
        corner and moving clockwise.
    """
    w = (
        spatial.distance.cdist(box[0][np.newaxis], box[1][np.newaxis], "euclidean")
        + spatial.distance.cdist(box[2][np.newaxis], box[3][np.newaxis], "euclidean")
    ) / 2
    h = (
        spatial.distance.cdist(box[0][np.newaxis], box[3][np.newaxis], "euclidean")
        + spatial.distance.cdist(box[1][np.newaxis], box[2][np.newaxis], "euclidean")
    ) / 2
    return int(w[0][0]), int(h[0][0])