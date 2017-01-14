from __future__ import print_function
import cv2
from deepflow2 import deepflow2
from deepmatching import deepmatching
import os
import numpy as np
import fisheye
import fisheye_old
from math import *

number_points = 32 ** 2
output_dimension = int(sqrt(number_points))
image_dimension = 900

fov = 158.0
camera_f = 47.6083  # blender calculated value

i2 = image_dimension / 2

# dimension of cropped image
nx0, ny0 = fisheye_old.new_c(i2, 0, camera_f, i2, i2)
nx2, ny2 = fisheye_old.new_c(i2, image_dimension, camera_f, i2, i2)
crop_dimension = ny2 - ny0

f = fisheye.focal_distance(fov=fov, image_dimension=crop_dimension)

path = "../input/"
pathForLua = "input/"
point_file = open('points.txt', 'r')
imageNames = [[open(path + 'inputImagesLeft' +
                    str(i) + '.txt', 'w') for i in xrange(4)],
              [open(path + 'inputImagesRight' +
                    str(i) + '.txt', 'w') for i in xrange(4)]]

def convert_images(ims, left):
    if left:
        tmp = [i[:, 0:i.shape[1] / 2] for i in ims]
    else:
        tmp = [i[:, i.shape[1] / 2:i.shape[1]] for i in ims]
    """tmp = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in tmp]"""
    tmp = [cv2.GaussianBlur(i,(5,5),0) for i in tmp]
    return tmp


# load coordinates of receptors
points = fisheye.load_sphere_coordinates(point_file, number_points)
# create distortion maps
# map for distorting image before flow calculation
dist_map_x, dist_map_y = fisheye_old.calculate_maps(image_dimension, camera_f)
# map from retina coordinate to distorted image
retina_map_x, retina_map_y = fisheye.calculate_maps(
    points, crop_dimension, fov=fov, camera_f=camera_f)
cv2.imwrite('map_on_distorted.png',
            fisheye.show_map(retina_map_x, retina_map_y, crop_dimension, crop_dimension))


# traverse folders
folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
folders = [folder + '/' for folder in folders]
folders = sorted(folders)

sides = ['left', 'right']
for k in xrange(len(folders)):
    try:
        print(str(k / float(len(folders)) * 100) + '%')
        folder = folders[k]
        print(folder)
        files = os.listdir(path + folder)
        files = sorted(files)
        # load images
        q = 0
        images = [cv2.imread(path + folder + fl, 1)
                  for fl in files if str(q) + '.png' == fl or str(q + 1) + '.png' == fl]

        # this will be saved as input for the net
        out = [[np.zeros((output_dimension, output_dimension))
                for x in xrange(4)],
               [np.zeros((output_dimension, output_dimension))
                for x in xrange(4)]]

        for s in xrange(len(sides)):
            ims = convert_images(images, left=(s == 0))
            ims = [fisheye_old.distort(i, dist_map_x, dist_map_y, camera_f)
                   for i in ims]

            #cv2.imwrite("map_on_distorted.png", fisheye.show_map(
            #    retina_map_x, retina_map_y, ims[0].shape[0], ims[0].shape[1]))

            for i in xrange(len(ims)):
                cv2.imwrite(path + folder + "converted_" +
                            sides[s] + str(i) + ".png", ims[i])

            # try:
            """allFlow2 = cv2.calcOpticalFlowFarneback(
                prev=ims[0], next=ims[1], flow=None,
                pyr_scale=0.8, levels=3, winsize=15,
                iterations=3, poly_n=3, poly_sigma=1.2, flags=0)"""


            """allFlow = np.zeros_like(ims[0]),
            cv2.calcOpticalFlowSF(
                ims[0], ims[1], flow=allFlow, sigma_dist=4.1,
                layers=3, averaging_block_size=2, max_flow=4,
                sigma_color=25.5, postprocess_window=18,
                sigma_dist_fix=55.0, sigma_color_fix=25.5, occ_thr=0.35,
                upscale_averaging_radius=18, upscale_sigma_dist=55.0,
                upscale_sigma_color=25.5, speed_up_thr=10)"""

            matches = deepmatching(ims[0], ims[1])
            allFlow = deepflow2(ims[0], ims[1], matches, '-sintel')  # -sintel -middlebury -kitti

            # apply blur
            # allFlow = cv2.GaussianBlur(allFlow,(9,9),0)

            # display flow
            """
            hsv = np.zeros((allFlow.shape[0], allFlow.shape[1], 3), np.uint8)
            mag, ang = cv2.cartToPolar(allFlow[...,0], allFlow[...,1])
            hsv[...,0] = ang * 180 / np.pi / 2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            cv2.imshow('flow', bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""

            # blacks for flow
            flow = [np.zeros((allFlow.shape[0], allFlow.shape[1])).astype(float)
                    for x in xrange(4)]

            for axis in xrange(4):
                # axis 0 up, 1 right, 2 down, 3 left
                # therefore axis % 2 == 0 for y axis, axis >= 2 for negative
                # values
                a = min if axis >= 2 else max
                for x in xrange(flow[axis].shape[1]):
                    for y in xrange(flow[axis].shape[0]):
                        flow[axis][y, x] = abs(a(0, allFlow[y, x][axis % 2]))
                out[s][axis] = fisheye.distort(
                    flow[axis], out[s][axis], retina_map_x, retina_map_y)

            #fisheye.show_vector_field(out[s], points_on_sphere=points)

        # regularize both eyes together and save images
        maxm = 0.0
        minm = 100000.0
        for s in xrange(len(sides)):
            for axis in xrange(4):
                maxm = max(maxm, np.max(out[s][axis]))
                minm = min(minm, np.min(out[s][axis]))

        for s in xrange(len(sides)):
            for axis in xrange(4):
                out[s][axis] = ((out[s][axis] - minm) /
                                (maxm - minm) * 255).astype('uint8')
                name = folder + 'out_' + sides[s] + str(axis) + '.png'
                cv2.imwrite(path + name, out[s][axis])
                print(pathForLua + name, file=imageNames[s][axis])

                """cv2.imshow(name, flow[axis])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                fisheye.show_sphere(out[s][axis], points_on_sphere=points)"""


    except IOError:
        print('Error on folder ' + folders[k])
