import tensorflow as tf

from tensorflow import keras
import numpy as np
import cv2

from utils import read


def build_keras_model(weights_path = None, backbone_name="vgg"):
    inputs = keras.layers.Input((None, None, 3))

    if backbone_name == "vgg":
        s1, s2, s3, s4 = build_vgg_backbone(inputs)
    else:
        raise NotImplementedError

    s5 = keras.layers.MaxPooling2D(
        pool_size=3, strides=1, padding="same", name="basenet.slice5.0"
    )(s4)
    s5 = keras.layers.Conv2D(
        1024,
        kernel_size=(3, 3),
        padding="same",
        strides=1,
        dilation_rate=6,
        name="basenet.slice5.1",
    )(s5)
    s5 = keras.layers.Conv2D(
        1024, kernel_size=1, strides=1, padding="same", name="basenet.slice5.2"
    )(s5)

    y = keras.layers.Concatenate()([s5, s4])
    y = upconv(y, n=1, filters=512)
    y = UpsampleLike()([y, s3])
    y = keras.layers.Concatenate()([y, s3])
    y = upconv(y, n=2, filters=256)
    y = UpsampleLike()([y, s2])
    y = keras.layers.Concatenate()([y, s2])
    y = upconv(y, n=3, filters=128)
    y = UpsampleLike()([y, s1])
    y = keras.layers.Concatenate()([y, s1])
    features = upconv(y, n=4, filters=64)

    y = keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=1, padding="same", name="conv_cls.0"
    )(features)
    y = keras.layers.Activation("relu", name="conv_cls.1")(y)
    y = keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=1, padding="same", name="conv_cls.2"
    )(y)
    y = keras.layers.Activation("relu", name="conv_cls.3")(y)
    y = keras.layers.Conv2D(
        filters=16, kernel_size=3, strides=1, padding="same", name="conv_cls.4"
    )(y)
    y = keras.layers.Activation("relu", name="conv_cls.5")(y)
    y = keras.layers.Conv2D(
        filters=16, kernel_size=1, strides=1, padding="same", name="conv_cls.6"
    )(y)
    y = keras.layers.Activation("relu", name="conv_cls.7")(y)
    y = keras.layers.Conv2D(
        filters=2, kernel_size=1, strides=1, padding="same", name="conv_cls.8"
    )(y)
    if backbone_name != "vgg":
        y = keras.layers.Activation("sigmoid")(y)
    model = keras.models.Model(inputs=inputs, outputs=y)
    if weights_path is not None:
        if weights_path.endswith(".h5"):
            model.load_weights(weights_path)
        else:
            raise NotImplementedError(f"Cannot load weights from {weights_path}")
    return model

def upconv(x, n, filters):
    x = keras.layers.Conv2D(
        filters=filters, kernel_size=1, strides=1, name=f"upconv{n}.conv.0"
    )(x)
    x = keras.layers.BatchNormalization(
        epsilon=1e-5, momentum=0.9, name=f"upconv{n}.conv.1"
    )(x)
    x = keras.layers.Activation("relu", name=f"upconv{n}.conv.2")(x)
    x = keras.layers.Conv2D(
        filters=filters // 2,
        kernel_size=3,
        strides=1,
        padding="same",
        name=f"upconv{n}.conv.3",
    )(x)
    x = keras.layers.BatchNormalization(
        epsilon=1e-5, momentum=0.9, name=f"upconv{n}.conv.4"
    )(x)
    x = keras.layers.Activation("relu", name=f"upconv{n}.conv.5")(x)
    return x

def build_vgg_backbone(inputs):
    x = make_vgg_block(inputs, filters=64, n=0, pooling=False, prefix="basenet.slice1")
    x = make_vgg_block(x, filters=64, n=3, pooling=True, prefix="basenet.slice1")
    x = make_vgg_block(x, filters=128, n=7, pooling=False, prefix="basenet.slice1")
    x = make_vgg_block(x, filters=128, n=10, pooling=True, prefix="basenet.slice1")
    x = make_vgg_block(x, filters=256, n=14, pooling=False, prefix="basenet.slice2")
    x = make_vgg_block(x, filters=256, n=17, pooling=False, prefix="basenet.slice2")
    x = make_vgg_block(x, filters=256, n=20, pooling=True, prefix="basenet.slice3")
    x = make_vgg_block(x, filters=512, n=24, pooling=False, prefix="basenet.slice3")
    x = make_vgg_block(x, filters=512, n=27, pooling=False, prefix="basenet.slice3")
    x = make_vgg_block(x, filters=512, n=30, pooling=True, prefix="basenet.slice4")
    x = make_vgg_block(x, filters=512, n=34, pooling=False, prefix="basenet.slice4")
    x = make_vgg_block(x, filters=512, n=37, pooling=False, prefix="basenet.slice4")
    x = make_vgg_block(x, filters=512, n=40, pooling=True, prefix="basenet.slice4")
    vgg = keras.models.Model(inputs=inputs, outputs=x)
    return [
        vgg.get_layer(slice_name).output
        for slice_name in [
            "basenet.slice1.12",
            "basenet.slice2.19",
            "basenet.slice3.29",
            "basenet.slice4.38",
        ]
    ]

def make_vgg_block(x, filters, n, prefix, pooling=True):
    x = keras.layers.Conv2D(
        filters=filters,
        strides=(1, 1),
        kernel_size=(3, 3),
        padding="same",
        name=f"{prefix}.{n}",
    )(x)
    x = keras.layers.BatchNormalization(
        momentum=0.1, epsilon=1e-5, axis=-1, name=f"{prefix}.{n+1}"
    )(x)
    x = keras.layers.Activation("relu", name=f"{prefix}.{n+2}")(x)
    if pooling:
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2), padding="valid", strides=(2, 2), name=f"{prefix}.{n+3}"
        )(x)
    return x

def compute_input(image):
    # should be RGB order
    image = image.astype("float32")
    mean = np.array([0.485, 0.456, 0.406])
    variance = np.array([0.229, 0.224, 0.225])

    image -= mean * 255
    image /= variance * 255
    return image

def getBoxes(
    y_pred,
    detection_threshold=0.7,
    text_threshold=0.4,
    link_threshold=0.4,
    size_threshold=10,
):
    box_groups = []
    for y_pred_cur in y_pred:
        # Prepare data
        textmap = y_pred_cur[..., 0].copy()
        linkmap = y_pred_cur[..., 1].copy()
        img_h, img_w = textmap.shape

        _, text_score = cv2.threshold(
            textmap, thresh=text_threshold, maxval=1, type=cv2.THRESH_BINARY
        )
        _, link_score = cv2.threshold(
            linkmap, thresh=link_threshold, maxval=1, type=cv2.THRESH_BINARY
        )
        n_components, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.clip(text_score + link_score, 0, 1).astype("uint8"), connectivity=4
        )
        boxes = []
        for component_id in range(1, n_components):
            # Filter by size
            size = stats[component_id, cv2.CC_STAT_AREA]

            if size < size_threshold:
                continue

            # If the maximum value within this connected component is less than
            # text threshold, we skip it.
            if np.max(textmap[labels == component_id]) < detection_threshold:
                continue

            # Make segmentation map. It is 255 where we find text, 0 otherwise.
            segmap = np.zeros_like(textmap)
            segmap[labels == component_id] = 255
            segmap[np.logical_and(link_score, text_score)] = 0
            x, y, w, h = [
                stats[component_id, key]
                for key in [
                    cv2.CC_STAT_LEFT,
                    cv2.CC_STAT_TOP,
                    cv2.CC_STAT_WIDTH,
                    cv2.CC_STAT_HEIGHT,
                ]
            ]

            # Expand the elements of the segmentation map
            niter = int(np.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, sy = max(x - niter, 0), max(y - niter, 0)
            ex, ey = min(x + w + niter + 1, img_w), min(y + h + niter + 1, img_h)
            segmap[sy:ey, sx:ex] = cv2.dilate(
                segmap[sy:ey, sx:ex],
                cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter)),
            )

            # Make rotated box from contour
            contours = cv2.findContours(
                segmap.astype("uint8"),
                mode=cv2.RETR_TREE,
                method=cv2.CHAIN_APPROX_SIMPLE,
            )[-2]
            contour = contours[0]
            box = cv2.boxPoints(cv2.minAreaRect(contour))

            # Check to see if we have a diamond
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = contour[:, 0, 0].min(), contour[:, 0, 0].max()
                t, b = contour[:, 0, 1].min(), contour[:, 0, 1].max()
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
            else:
                # Make clock-wise order
                box = np.array(np.roll(box, 4 - box.sum(axis=1).argmin(), 0))
            boxes.append(2 * box)
        box_groups.append(np.array(boxes))
    return box_groups


class UpsampleLike(keras.layers.Layer):
    """Keras layer for upsampling a Tensor to be the same shape as another Tensor."""

    # pylint:disable=unused-argument
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        if keras.backend.image_data_format() == "channels_first":
            raise NotImplementedError
        else:
            # pylint: disable=no-member
            return tf.compat.v1.image.resize_bilinear(
                source, size=(target_shape[1], target_shape[2]), half_pixel_centers=True
            )

    def compute_output_shape(self, input_shape):
        if keras.backend.image_data_format() == "channels_first":
            raise NotImplementedError
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)

class Detector:
    """A text detector using the CRAFT architecture.

    Args:
        optimizer: The optimizer to use for training the model.
        backbone_name: The backbone to use. Currently, only 'vgg' is supported.
    """

    def __init__(
        self,
        optimizer="adam",
        backbone_name="vgg",
    ):
        self.model = build_keras_model(
            weights_path="detection.h5", backbone_name=backbone_name
        )
        self.model.compile(loss="mse", optimizer=optimizer)

    def detect(
        self,
        images,
        detection_threshold=0.7,
        text_threshold=0.4,
        link_threshold=0.4,
        size_threshold=10,
        **kwargs,
    ):
        """Recognize the text in a set of images.

        Args:
            images: Can be a list of numpy arrays of shape HxWx3 or a list of
                filepaths.
            link_threshold: This is the same as `text_threshold`, but is applied to the
                link map instead of the text map.
            detection_threshold: We want to avoid including boxes that may have
                represented large regions of low confidence text predictions. To do this,
                we do a final check for each word box to make sure the maximum confidence
                value exceeds some detection threshold. This is the threshold used for
                this check.
            text_threshold: When the text map is processed, it is converted from confidence
                (float from zero to one) values to classification (0 for not text, 1 for
                text) using binary thresholding. The threshold value determines the
                breakpoint at which a value is converted to a 1 or a 0. For example, if
                the threshold is 0.4 and a value for particular point on the text map is
                0.5, that value gets converted to a 1. The higher this value is, the less
                likely it is that characters will be merged together into a single word.
                The lower this value is, the more likely it is that non-text will be detected.
                Therein lies the balance.
            size_threshold: The minimum area for a word.
        """
        images = [compute_input(read(image)) for image in images]
        boxes = getBoxes(
            self.model.predict(np.array(images), **kwargs),
            detection_threshold=detection_threshold,
            text_threshold=text_threshold,
            link_threshold=link_threshold,
            size_threshold=size_threshold,
        )
        return boxes