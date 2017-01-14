import cv2
import numpy as np
from math import *

def polar2z(r, theta):
    return r * np.exp( 1j * theta)

def z2polar(z):
    return (np.abs(z), np.angle(z))

def correct_vector(x,y,dx,dy,cx,cy):
    theta = 1  # sqrt((x - cx) ** 2 + (y - cy) ** 2)
    (r, phi) = z2polar((x - cx) + (y - cy) * 1j)
    dphi = (sin(phi) * dx - cos(phi) * dy) / -theta
    dtheta = cos(phi) * dx + sin(phi) * dy
    return dphi, dtheta

# location of pixel in new picture respective to location in old picture
def new_cOLD(x, y, f, cx, cy):
    (r, alpha) = z2polar((x - cx) + (y - cy) * 1j)
    new_r = 2 * f * sin(atan(r / f) / 2)
    new_x = np.real(polar2z(new_r, alpha))
    new_y = np.imag(polar2z(new_r, alpha))
    # print(x,new_x + cx)
    return int(round(new_x + cx)), int(round(new_y + cy))

def new_c(x, y, f, cx, cy):
    (r, alpha) = z2polar((x - cx) + (y - cy) * 1j)
    new_r = f * atan(r / f)
    new_x = np.real(polar2z(new_r, alpha))
    new_y = np.imag(polar2z(new_r, alpha))
    return int(round(new_x + cx)), int(round(new_y + cy))

# location of pixel in old picture respective to location in new picture
def old_cOLD(x, y, f, cx, cy):
    (r, alpha) = z2polar((x - cx) + (y - cy) * 1j)
    try:
        # new_r = tan(2 * asin(r / (2 * f))) * f
        new_r = r * sqrt(1.0 - (r / f) ** 2.0 / 4.0) / (1.0 - (r / f) ** 2.0 / 2.0)
    except ValueError:
        new_r = 0

    new_x = np.real(polar2z(new_r, alpha))
    new_y = np.imag(polar2z(new_r, alpha))
    return int(round(new_x + cx)), int(round(new_y + cy))

def old_c(x, y, f, cx, cy):
    (r, alpha) = z2polar((x - cx) + (y - cy) * 1j)
    new_r = f * tan(r / f)
    new_x = np.real(polar2z(new_r, alpha))
    new_y = np.imag(polar2z(new_r, alpha))
    return int(round(new_x + cx)), int(round(new_y + cy))

def calculate_maps(image_dimension, f=200):
    centerx = image_dimension / 2
    centery = image_dimension / 2

    map_x = np.zeros((image_dimension,image_dimension),np.float32)
    map_y = np.zeros((image_dimension,image_dimension),np.float32)

    # calculate new coordinate for each pixel
    for x in xrange(image_dimension):
        for y in xrange(image_dimension):
            # new_x, new_y = new_c(x, y, f, centerx, centery)
            # map_x.itemset((new_y,new_x), x)
            # map_y.itemset((new_y,new_x), y)
            old_x, old_y = old_c(x, y, f, centerx, centery)
            map_y.itemset((y,x), old_y)
            map_x.itemset((y,x), old_x)

    return map_x, map_y

def distort(img, map_x, map_y, f=200):
    centerx = img.shape[0] / 2
    centery = img.shape[1] / 2

    res = cv2.remap(img,map_x,map_y,cv2.INTER_LINEAR)

    # crop result
    nx0, ny0 = new_c(res.shape[1] / 2, 0, f, centery, centerx)
    nx1, ny1 = new_c(0, res.shape[0] / 2, f, centery, centerx)
    nx2, ny2 = new_c(res.shape[1] / 2, res.shape[0], f, centery, centerx)
    nx3, ny3 = new_c(res.shape[1], res.shape[0] / 2, f, centery, centerx)
    # print(ny0,ny2,nx1,nx3)
    res = res[ny0:ny2, nx1:nx3]

    return res
