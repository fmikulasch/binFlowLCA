import numpy as np
from math import *

s = [lambda x, y, f: sqrt(f**2 + y**2) / sqrt(f**2 + x**2 + y**2),
     lambda x, y, f: sqrt(f**2 + x**2) / sqrt(f**2 + x**2 + y**2),
     lambda x, y, f: sqrt(f**2 + y**2) / sqrt(f**2 + x**2 + y**2),
     lambda x, y, f: sqrt(f**2 + x**2) / sqrt(f**2 + x**2 + y**2)]

def create_regularize_maps(image_dimension, f):
    image_dimension = int(image_dimension)
    maps = []
    for i in xrange(4):
        maps.append(np.zeros((image_dimension,image_dimension)))
        for x in xrange(image_dimension):
            for y in xrange(image_dimension):
                maps[i][y,x] = s[i](x - image_dimension / 2,y - image_dimension / 2,f)
    return maps

def regularize_flow(flow,maps,axis):
    return np.multiply(flow,maps[axis])
