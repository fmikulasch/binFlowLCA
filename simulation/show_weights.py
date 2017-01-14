import fisheye
import cv2
import sys
import matplotlib.pyplot as plt

point_file = open('points.txt', 'r')
number_points = 32 ** 2

weight_files = [sys.argv[i] for i in range(1,9)]
image_dimension = 32
points = fisheye.load_sphere_coordinates(point_file, number_points)

def cut_image(image):
    image_x = image.shape[1] / image_dimension
    image_y = image.shape[0] / image_dimension
    res = []
    for x in xrange(image_x):
        for y in xrange(image_y):
            weight = image[y * image_dimension:(y + 1) * image_dimension,
                           x * image_dimension:(x + 1) * image_dimension]
            res.append(weight)
            """cv2.imshow("d",weight)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""
    return res

weights_list = []
for weight_file in weight_files:
    image = cv2.imread(weight_file, 0)
    weights_list.append(cut_image(image))

for i in xrange(len(weights_list[0])):
    whole_weights = []
    for weights in weights_list:
        whole_weights.append(weights[i])
    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(16, 8, forward=True)
    fisheye.plot_vector_field(whole_weights[0:4], points_on_sphere=points, ax=axs[0])
    fisheye.plot_vector_field(whole_weights[4:8], points_on_sphere=points, ax=axs[1])
    plt.draw()
    plt.show()
