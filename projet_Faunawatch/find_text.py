import recognition
import detection
import numpy as np

from utils import read, resize_image, pad, adjust_boxes



class Find_Text:
    """A wrapper for a combination of detector and recognizer.

    Args:
        scale: The scale factor to apply to input images
        max_size: The maximum single-side dimension of images for
            inference.
    """

    def __init__(self, scale=2, max_size=2048):
        
        self.scale = scale
        self.detector = detection.Detector()
        self.recognizer = recognition.Recognizer()
        self.max_size = max_size

    def recognize(self, images):
        """Run the pipeline on one or multiples images.

        Args:
            images: The images to parse (can be a list of actual images or a list of filepaths)

        Returns:
            A list of lists of (text, box) tuples.
        """

        # Make sure we have an image array to start with.
        if not isinstance(images, np.ndarray):
            images = [read(image) for image in images]
        # This turns images into (image, scale) tuples temporarily
        images = [
            resize_image(image, max_scale=self.scale, max_size=self.max_size)
            for image in images
        ]
        max_height, max_width = np.array(
            [image.shape[:2] for image, scale in images]
        ).max(axis=0)
        scales = [scale for _, scale in images]
        images = np.array(
            [
                pad(image, width=max_width, height=max_height)
                for image, _ in images
            ]
        )
        detection_kwargs = {}
        recognition_kwargs = {}
        
        box_groups = self.detector.detect(images=images, **detection_kwargs)
        prediction_groups = self.recognizer.recognize_from_boxes(
            images=images, box_groups=box_groups, **recognition_kwargs
        )
        box_groups = [
            adjust_boxes(boxes=boxes, boxes_format="boxes", scale=1 / scale)
            if scale != 1
            else boxes
            for boxes, scale in zip(box_groups, scales)
        ]
        return [
            list(zip(predictions, boxes))
            for predictions, boxes in zip(prediction_groups, box_groups)
        ]