import math

import cv2
import numpy as np


def unit_vector(v):
    """Returns the unit vector of the vector."""
    return v / np.linalg.norm(v)


def get_angle_between_vectors(v1, v2):
    """Define angle between two vectors. Outpur angle always positive."""
    radian = np.arccos(np.dot(unit_vector(v1), unit_vector(v2)))
    return math.degrees(radian)


def get_angle_by_fitline(contour):
    """Get angle of contour using cv2.fitLine."""
    # Output line parameters. In case of 2D fitting, it is a vector of 4-elements vector [vx,vy, x0,y0],
    # where [vx,vy] is a normalized vector collinear to the line and [x0,y0] is a point on the line.
    vx, vy, x, y = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    v1 = np.array([vx[0], vy[0]])
    v2 = np.array([1, 0])
    angle = get_angle_between_vectors(v1, v2)
    # get_line_angle return angle between vectors always positive
    # so multiply by minus one if the line is negative
    if vy > 0:
        angle *= -1
    return angle


def get_angle_by_minarearect(contour):
    """Get angle of contour using cv2.minAreaRect."""
    rotated_rectangle = cv2.minAreaRect(contour)
    angle = rotated_rectangle[2]
    angle *= -1  # take perpendicular angle to the bottom rectangle side
    # take the opposite angle if the rectangle is too rotated
    if angle < -45:
        angle += 90
    return angle


def get_image_angle(contours, use_fitline=True):
    """Define the angle of the image using the contours of the words."""
    angles = [get_angle_by_fitline(contour)
              if use_fitline
              else get_angle_by_minarearect(contour)
              for contour in contours]
    return np.median(np.array(angles))


def rotate_image_and_contours(image, contours, angle):
    """Rotate the image and contours by the angle."""
    rotated_image, rotation_mat = rotate_image(image, angle)
    rotated_contours = [cv2.transform(contour, rotation_mat) for contour in contours]
    return rotated_image, rotated_contours


def rotate_image(mat, angle):
    """
    https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat, rotation_mat


class ImageAngleRestorer:
    """Define the angle of the image and rotates the image and contours to
    this angle.

    Args:
        min_angle_to_rotate (int): The safe range of angles within which image
            rotation does not occur (-min_angle_to_rotate; min_angle_to_rotate)
    """

    def __init__(self, args, min_angle_to_rotate=0.5):
        self.restoring_class_names = args['restoring_class_names']
        self.min_angle_to_rotate = min_angle_to_rotate

    def __call__(self, image, data):
        contours = []
        restoring_contours = []
        for prediction in data['predictions']:
            contour = np.array([prediction['polygon']])
            contours.append(contour)
            if prediction['class_name'] in self.restoring_class_names:
                restoring_contours.append(contour)

        angle = get_image_angle(restoring_contours)
        if abs(angle) > self.min_angle_to_rotate:
            image, contours = rotate_image_and_contours(image, contours, -angle)

        for prediction, contour in zip(data['predictions'], contours):
            contour = [[int(p[0]), int(p[1])] for p in contour[0]]
            prediction['rotated_polygon'] = contour
        return image, data
