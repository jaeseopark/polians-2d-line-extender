import cv2
import numpy as np
import os


def find_circles(binary_image):
    cueball_icon_radius = max(binary_image.shape) // 50
    gradient_threshold = 10
    accumulator_threshold = 30
    inverse_accumulator_resolution = 1.5
    circles = cv2.HoughCircles(
        binary_image,
        cv2.HOUGH_GRADIENT,
        inverse_accumulator_resolution,
        minDist=10,
        param1=gradient_threshold,
        param2=accumulator_threshold,
        minRadius=cueball_icon_radius // 3,
        maxRadius=cueball_icon_radius
    )

    return np.uint16(np.around(circles))[0]


def find_lines(binary_image):
    minLineLength = max(binary_image.shape) // 65
    maxLineGap = 5
    return cv2.HoughLinesP(binary_image, 1, np.pi / 180, 40,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)


def draw_output(image_to_paint_over, circles, lines):
    for line in lines:
        for (x1, y1, x2, y2) in line:
            cv2.line(image_to_paint_over, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for (x, y, r) in circles:
        cv2.circle(image_to_paint_over, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(image_to_paint_over, (x - 5, y - 5),
                      (x + 5, y + 5), (0, 128, 255), -1)


def display(img, shape, target_dimention):
    longer_side = max(shape)
    factor = target_dimention / longer_side

    if factor < 1:
        # 'shape' returns height first but resize() expects width first. Reverse the tuple.
        size = tuple(reversed(tuple(ti // 2 for ti in shape)))
        img = cv2.resize(img, size)

    cv2.imshow('window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def imwrite(image, dir, filename, suffix):
    dir_output = os.path.join(dir, "output")
    if (not os.path.exists(dir_output)):
        os.mkdir(dir_output)

    cv2.imwrite(os.path.join(dir_output, filename + suffix), image)


def get_base_images(path):
    thresholding_value = 150
    ret, binary_image = cv2.threshold(
        cv2.imread(path, 0),
        thresholding_value,
        255,
        cv2.THRESH_BINARY
    )

    org_image = cv2.imread(path)

    return org_image, binary_image
