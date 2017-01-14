import cv2
import numpy as np
import regularize
from math import *


def gauss2d(x, y, theta, x0, y0, sigma_x, sigma_y):
    a = cos(theta) ** 2 / (2 * sigma_x ** 2) + \
        sin(theta) ** 2 / (2 * sigma_y ** 2)
    b = -sin(2 * theta) / (4 * sigma_x ** 2) + \
        sin(2 * theta) / (4 * sigma_y ** 2)
    c = sin(theta) ** 2 / (2 * sigma_x ** 2) + \
        cos(theta) ** 2 / (2 * sigma_y ** 2)
    A = 255

    Z = A * exp( - (a * (x - x0) ** 2 - 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2))
    return Z


def gaussian_for_position(x0, y0, image_dimension, f):
    mid = image_dimension / 2
    z = (x0 - mid) + (y0 - mid) * 1j
    r = np.abs(z)
    theta = np.angle(z)
    sigma_y = image_dimension / 10 / f * atan(r / f)
    sigma_x = image_dimension / 10

    res = np.zeros((image_dimension, image_dimension))
    for x in xrange(image_dimension):
        for y in xrange(image_dimension):
            res[y, x] = gauss2d(x, y, theta, x0, y0, sigma_x, sigma_y)
    return res
