import cv2
import numpy as np


def convert_to_optical_flow_image(prev, next, mask):
    """
    Converts two images to optical flow
    Args:
        prev: prev frame
        next: next frame
        mask: mask
    Returns:
        optical flow in rgb
    """

    if not (prev.mode == 'RGB') == 2:
        prev = np.array(prev)
        next = np.array(next)
        prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        next = cv2.cvtColor(next, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_image = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    return flow_image


def calculate_flow_mag_and_or(prev, next, mask):

    # Convert the images to grayscale
    prev = np.array(prev)
    next = np.array(next)
    gray1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

    # Compute the optical flow using the Farneback algorithm
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_image = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    flow_image = cv2.cvtColor(flow_image, cv2.COLOR_BGR2GRAY)

    # Convert the flow vectors to polar coordinates
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # # Scale the magnitude values to lie between 0 and 1
    magnitude_scaled = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

    # Scale the orientation values to lie between 0 and 255
    angle_scaled = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the orientation values to grayscale
    orientation = np.uint8(angle_scaled)

    return flow_image, magnitude_scaled, orientation

