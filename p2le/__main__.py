import cv2
import numpy as np
import os
from math import sqrt


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


def sanitize_and_extend(circles, lines, shape):
    distance_threshold = max(shape) / 200

    def get_distance(p1, p2):
        return sqrt(((p2[0] - p1[0])**2) + ((p2[1] - p1[1])**2))

    def is_close(x1, y1, x2, y2):
        return get_distance((x1, y1), (x2, y2)) < distance_threshold

    def extend(start, end, limit=None):
        # TODO: use the premiter of the pool table as the 'limit' and perform the line intersection logic.
        # For now, extend the line by x2.5
        x = start[0] + (end[0] - start[0]) * 2.5
        y = start[1] + (end[1] - start[1]) * 2.5
        return int(x), int(y)

    # Discard circles that are not in the center of the image.
    center_of_image = tuple(ti // 2 for ti in shape)
    circles = sorted(circles, key=lambda c: get_distance(
        center_of_image, c[:2]))[:len(circles) // 2]
    # for (cx, cy, cr) in circles:

    # closest_line

    lines_new = list()
    for line in lines:
        for (x1, y1, x2, y2) in line:
            for (cx, cy, cr) in circles:
                p1 = None
                if is_close(x1, y1, cx, cy):
                    x2, y2 = extend((x1, y1), (x2, y2))
                    lines_new.append([(x1, y1, x2, y2)])
                elif is_close(x2, y2, cx, cy):
                    x1, y1 = extend((x2, y2), (x1, y1))
                    lines_new.append([(x1, y1, x2, y2)])

    return circles, lines_new


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


def main(dir, filename, show=False):
    path = os.path.join(dir, filename)

    thresholding_value = 150
    ret, binary_image = cv2.threshold(cv2.imread(
        path, 0), thresholding_value, 255, cv2.THRESH_BINARY)

    org_image = cv2.imread(path)
    circles = find_circles(binary_image)
    lines = find_lines(binary_image)

    circles, lines = sanitize_and_extend(circles, lines, binary_image.shape)
    draw_output(org_image, circles, lines)

    imwrite(org_image, dir, filename, ".annotated.jpg")
    imwrite(binary_image, dir, filename, ".binary.jpg")

    if show:
        display(org_image, binary_image.shape, target_dimention=1500)


if __name__ == "__main__":
    # TODO: get args from the user. For now use the static file
    for filename in os.listdir("data"):
        if filename == "output":
            continue
        main("data", filename)
