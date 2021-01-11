import os
from math import sqrt
from p2le.cv2wrapper import imwrite, find_circles, find_lines, draw_output, get_base_images


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


def main(dir, filename, show=False):
    path = os.path.join(dir, filename)
    org_image, binary_image = get_base_images(path)

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
