import cv2
import numpy as np
from math import *
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import string
import fisheye_old


def load_sphere_coordinates(file, number_points):
    points = []

    lineno = 0
    while 1:
        s = file.readline()
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
    # sorting of sphere points (for z) stays this way
    points.sort(key=lambda x: x[2])
    # only keep points with lowest z-coordinates
    points = points[0:number_points]
    return points


def get_point_on_image(sphere_coordinate, f, image_dimension):
    x, y, z = sphere_coordinate
    t = f / z  # scaling of ray
    old_x = min((x * t + image_dimension / 2), image_dimension - 1)
    old_y = min((y * t + image_dimension / 2), image_dimension - 1)
    return old_x, old_y

def focal_distance(fov,image_dimension):
    # assuming the image is square, points_on_sphere are on the unit sphere
    # with len = 2^n
    fov = fov / 180.0 * pi

    # distance of imageplane from origin
    f = (image_dimension / 2.0) / tan(fov / 2.0)

    return f

def calculate_maps(points_on_sphere, crop_dimension, image_dimension=128.0, fov=160.0, camera_f=160):
    f = focal_distance(fov,crop_dimension) * 2

    # resulting square representation
    output_dim = int(sqrt(len(points_on_sphere)))

    map_x = np.zeros((output_dim, output_dim), np.float32)
    map_y = np.zeros((output_dim, output_dim), np.float32)

    # calculate image coordinate for each receptor
    cs = []
    for p in points_on_sphere:
        # get point on original image
        x,y = get_point_on_image(p, f, image_dimension)
        # from there get point on distorted image
        x,y = fisheye_old.new_c(x, y, camera_f, image_dimension / 2, image_dimension / 2)
        # consider crop
        c = (x + (crop_dimension - image_dimension) / 2,y + (crop_dimension - image_dimension) / 2)
        cs.append(c)

    for x in xrange(output_dim):
        for y in xrange(output_dim):
            # map contains coordinates of pixels on the distorted image
            old_x, old_y = cs[x * output_dim + y]
            map_x.itemset((y, x), old_x)
            map_y.itemset((y, x), old_y)

    return map_x, map_y


def distort(src, dst, map_x, map_y):

    # print(map_y.type())
    res = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, dst)

    return res


def show_sphere(dst, points_on_sphere):
    output_dim = int(sqrt(len(points_on_sphere)))

    x = np.array([c[0] for c in points_on_sphere])
    y = np.array([c[1] for c in points_on_sphere])
    z = np.array([c[2] for c in points_on_sphere])

    conv = matplotlib.colors.ColorConverter()
    colors = [conv.to_rgb('0.0')] * len(points_on_sphere)
    for i in xrange(output_dim):
        for j in xrange(output_dim):
            colors[output_dim * i + j] = conv.to_rgb(str(dst[j,i] / 255.0))

    fig = plt.figure()
    ax = fig.add_subplot(111)  # , projection='3d')
    ax.set_axis_bgcolor('black')
    ax.scatter(-x, y, edgecolors='none', facecolors=np.array(colors))
    fig.set_size_inches(4, 4, forward=True)
    #fig.savefig('points.png', dpi=100)
    plt.show()

def show_vector_field(dsts, points_on_sphere):
    fig = plt.figure()
    ax = plt.gca()
    plot_vector_field(dsts, points_on_sphere, ax)
    #fig.set_size_inches(4, 4, forward=True)
    #fig.savefig('points.png', dpi=100)
    plt.draw()
    plt.show()

def plot_vector_field(dsts, points_on_sphere, ax):
    output_dim = int(sqrt(len(points_on_sphere)))

    x = np.array([c[0] for c in points_on_sphere])
    y = np.array([c[1] for c in points_on_sphere])
    z = np.array([c[2] for c in points_on_sphere])

    u = np.zeros(len(points_on_sphere))
    v = np.zeros(len(points_on_sphere))
    m = 255.0 * 100
    for i in xrange(len(dsts)):
        dsts[i] = dsts[i].astype('float')
    for i in xrange(output_dim):
        for j in xrange(output_dim):
            # axis 0 up, 1 right, 2 down, 3 left
            v[output_dim * i + j] = (dsts[1][j,i] - dsts[3][j,i]) / m
            u[output_dim * i + j] = (dsts[0][j,i] - dsts[2][j,i]) / m

    ax.quiver(-x,y,u,-v,angles='xy',scale_units='xy',scale=0.1)


def show_map(map_x, map_y, dimx, dimy):
    xs, ys = map_x.shape
    out = np.zeros((int(dimx), int(dimy)))
    for x in xrange(xs):
        for y in xrange(ys):
            out[int(round(map_y[y, x])), int(round(map_x[y, x]))] = 255
    # number of pixels is 1024: print(len([0 for o in out for p in o if p == 255]))
    return out
