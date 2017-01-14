import cv2
import numpy as np
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_sphere_coordinates(file, number_points):
    points = []

    lineno = 0
    while 1:
        s = infile.readline()
        if s == "":
            break
        sl = string.split(s)
        lineno = lineno + 1
        if len(sl) != 3:
            sys.stderr.write("line %d: expected three fields per line, got %d\n"
                             % (lineno, len(sl)))
            continue
        points.append(
            (string.atof(sl[0]), string.atof(sl[1]), string.atof(sl[2])))
    file.close()

    points.sort(key=lambda x: x[2])
    points = points[0:number_points]
    return points


def get_point_on_image(sphere_coordinate, f):
    x, y, z = sphere_coordinate
    t = f / z  # scaling of ray
    return round(x * t), round(y * t)


def calculate_map(points_on_sphere, image_dimension=128, fov=160):
    # assuming the image is square, points_on_sphere are on the unit sphere
    # with len = 2^n
    fov = fov / 180 * pi

    # distance of imageplane from origin
    f = (image_dimension / 2) / tan(fov / 2)

    # resulting square representation
    output_dim = sqrt(len(points_on_sphere))

    map = [np.zeros((image_dimension,image_dimension), np.float32)] * len(points_on_sphere)

    # calculate image coordinate for each receptor
    cs = []
    for p in points_on_sphere:
        cs.append(get_point_on_image(p, f))

    cs.sort(key=lambda c=c[0] * output_dim + c[1])

    for x in xrange(output_dim):
        for y in xrange(output_dim):
            old_x, old_y = cs[x * output_dim + y]
            map_x.itemset((y, x), old_x)
            map_y.itemset((y, x), old_y)

    return map_x, map_y


def distort(src, dst, map_x, map_y):
    # multiply and calc mean
    res = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, dst)

    return res

def show_sphere(dst, points_on_sphere):
    output_dim = sqrt(len(points_on_sphere))

    x = np.array([c[0] for c in dst])
    y = np.array([c[1] for c in dst])
    z = np.array([c[2] for c in dst])
    colors = np.zeros(len(points_on_sphere))
    for x in xrange(output_dim):
        for y in xrange(output_dim):
            colors[output_dim * x + y] = dst[x,y]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', facecolors=colors)
    plt.show()
