import cv2
import numpy as np


def clahe1channel(gray):
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    cl = np.expand_dims(cl, axis=2)
    return cl


def clahe3channel(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = np.array(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


def grayscale(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = np.expand_dims(g, axis=2)
    return g


def bilateral(bgr):
    b = cv2.bilateralFilter(bgr, 9, 75, 75)
    if len(b.shape) == 2:
        b = np.expand_dims(b, axis=2)
    return b


def gaussian(bgr):
    blur = cv2.GaussianBlur(bgr, (9, 9), 0)
    if len(blur.shape) == 2:
        blur = np.expand_dims(blur, axis=2)
    return blur
