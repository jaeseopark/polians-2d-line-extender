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


def find_lines(binary_image, min_radius):
    minLineLength = min_radius * 3
    maxLineGap = 5
    return cv2.HoughLinesP(binary_image, 1, np.pi / 180, 40,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)


def draw_output(image_to_paint_over, circles, lines):
    for line in lines:
        for x1, y1, x2, y2 in line:
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


def main(dir, filename, show=False):
    path = os.path.join(dir, filename)

    thresholding_value = 150
    ret, binary_image = cv2.threshold(cv2.imread(
        path, 0), thresholding_value, 255, cv2.THRESH_BINARY)

    org_image = cv2.imread(path)
    circles = find_circles(binary_image)
    min_radius = min([c[2] for c in circles])
    lines = find_lines(binary_image, min_radius)

    draw_output(org_image, circles, lines)

    cv2.imwrite(
        os.path.join(
            dir,
            "output",
            filename +
            ".binary.jpg"),
        binary_image)
    cv2.imwrite(
        os.path.join(
            dir,
            "output",
            filename +
            ".annotated.jpg"),
        org_image)

    if show:
        display(org_image, binary_image.shape, target_dimention=1500)


if __name__ == "__main__":
    # TODO: get args from the user. For now use the static file
    for filename in os.listdir("data"):
        if filename == "output":
            continue
        main("data", filename)
