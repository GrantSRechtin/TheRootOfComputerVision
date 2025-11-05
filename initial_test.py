import cv2
import time
import numpy as np
import sympy
import math
import transform_eyrie_board
from operator import itemgetter
from itertools import combinations


class testing:

    def __init__(self):

        self.active = True
        self.cv_image = None

        self.i = 0

        self.vc = cv2.VideoCapture(0)  # Webcam w=640 h=480

        path_base = "board_with_table.png"
        self.base_img = cv2.imread(path_base)
        self.sw = len(self.base_img)
        self.sh = len(self.base_img[0])

        path = "52B-RootBaseFactionBoardwithComponents-Editv2-web.webp"
        self.img = cv2.imread(path)

        self.img = self.blur(self.img)

        # Card Slots
        # self.rl = 0
        # self.rh = 255
        # self.gl = 0
        # self.gh = 41
        # self.bl = 38
        # self.bh = 130

        # Birdsong
        self.birdsong = (70, 130, 150, 150, 200, 240)

        # Birdsong Init
        self.birdsong_init = (0, 155, 196, 82, 196, 240)

        # Main
        self.rl = 70
        self.rh = 150
        self.gl = 140
        self.gh = 200
        self.bl = 150
        self.bh = 240

        if self.vc.isOpened():
            self.cv_image = self.img  # self.vc.read()

    def loop_wrapper(self):
        """loops run_loop"""

        cv2.namedWindow("video_window", cv2.WINDOW_NORMAL)
        cv2.namedWindow("binary_window")
        cv2.createTrackbar(
            "red lower bound", "binary_window", self.rl, 255, self.set_rl
        )
        cv2.createTrackbar(
            "red upper bound", "binary_window", self.rh, 255, self.set_rh
        )
        cv2.createTrackbar(
            "green lower bound", "binary_window", self.gl, 255, self.set_gl
        )
        cv2.createTrackbar(
            "green upper bound", "binary_window", self.gh, 255, self.set_gh
        )
        cv2.createTrackbar(
            "blue lower bound", "binary_window", self.bl, 255, self.set_bl
        )
        cv2.createTrackbar(
            "blue upper bound", "binary_window", self.bh, 255, self.set_bh
        )

        while self.active:
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):

        if not self.cv_image is None:

            # Get initial binary image
            self.binary_image = cv2.inRange(
                self.cv_image, self.birdsong_init[0:3], self.birdsong_init[3:6]
            )
            # self.binary_image = cv2.inRange(
            #     self.cv_image, (self.rl, self.gl, self.bl), (self.rh, self.gh, self.bh))

            cv2.imshow('video_window', self.cv_image)
            # cv2.imshow('binary_window', self.binary_image)

            self.check_input()

    def check_input(self):
        key = cv2.waitKey(20)
        if key == 27:
            cv2.destroyAllWindows()
            self.vc.release()
            self.active = False
        elif key == 32:
            contours = self.create_binary_contours(self.binary_image)
            pts = self.find_birdsong_border(contours[0][1], self.birdsong)
            print(pts)
            # Br Bl Tl Tr
            transformed_corners = [
                (204.4, 276), (46, 276), (46, 258), (204.4, 258)]
            cv2.namedWindow("Original image")
            cv2.namedWindow("Transformed image")
            cv2.imshow("Original image", self.cv_image)
            trans_image = transform_eyrie_board.transform_board(
                pts, transformed_corners, self.cv_image)
            cv2.imshow("Transformed image", trans_image)

    def blur(self, img):
        h, w, channels = img.shape
        return cv2.resize(
            cv2.resize(img, (700, 550), interpolation=cv2.INTER_AREA), (w, h)
        )

    def create_binary_contours(self, binary_image, invert=False):

        if invert:
            binary_image = cv2.bitwise_not(binary_image)

        contour_list, hierarchy = cv2.findContours(
            binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        contour_areas = [cv2.contourArea(c) for c in contour_list]
        contours = [
            (contour_areas[i], contour_list[i]) for i in range(len(contour_list))
        ]
        contours = sorted(contours, key=itemgetter(0))[::-1]

        return contours

    def create_contours(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 30, 200)
        contour_list, hierarchy = cv2.findContours(
            edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contour_areas = [cv2.contourArea(c) for c in contour_list]
        contours = [
            (contour_areas[i], contour_list[i]) for i in range(len(contour_list))
        ]
        contours = sorted(contours, key=itemgetter(0))[::-1]

        return contours

    def find_birdsong_border(self, contour, c_range):
        xs = [c[0][0] for c in contour]
        ys = [c[0][1] for c in contour]
        b_xs = (min(xs), max(xs))
        b_ys = (max(ys), min(ys))
        b_width = abs(max(xs) - min(xs))
        b_height = abs(max(ys) - min(ys))
        b_cx = (int)(min(xs) + (b_width / 2))
        b_cy = (int)(min(ys) + (b_height / 2))

        w = len(self.cv_image)
        h = len(self.cv_image[0])

        circle = np.zeros((w, h), dtype=np.uint8)
        cv2.rectangle(
            circle,
            (min(xs) - 2 * b_width, min(ys) - b_height),
            (max(xs) + 2 * b_width, max(ys) + b_height),
            255,
            -1,
        )

        binary_image = cv2.inRange(
            self.cv_image,
            (c_range[0], c_range[1], c_range[2]),
            (c_range[3], c_range[4], c_range[5]),
        )

        birdsong_region = cv2.bitwise_and(binary_image, circle)
        contours = self.create_binary_contours(birdsong_region)

        blank_image = np.zeros((h, w, 3), np.uint8)
        cv2.drawContours(blank_image, [contours[0][1]], 0, (0, 0, 255), 1)

        app = appx_best_fit_ngon(blank_image)
        pts = np.array(app, dtype=np.int32)
        # if pts.ndim == 2 and pts.shape[1] == 2:
        #     pts = pts.reshape((-1, 1, 2))

        for i in range(3):
            for pt in pts:
                pt = move_closest_point_towards(pt, contours[0][1])

        # pts[2][1] += 3
        # pts[3][1] += 3
        # pts[3][0] -= 3

        contour = self.cv_image.copy()
        cv2.drawContours(contour, [contours[0][1]], 0, (0, 0, 255), 1)

        result = self.cv_image.copy()
        cv2.polylines(result, [pts], True, (0, 0, 255), 2)

        cv2.imshow("CONTOUR", contour)
        cv2.imshow("QUAD", result)

        return pts

    def set_border(self, xs, ys):
        print(xs[0], xs[1])
        print(ys[1], ys[0])
        self.cv_image = self.cv_image[ys[1]: ys[0], xs[0]: xs[1]]

    def set_rl(self, val):
        """A callback function to handle the OpenCV slider to select the red lower bound"""
        self.rl = val

    def set_rh(self, val):
        """A callback function to handle the OpenCV slider to select the red upper bound"""
        self.rh = val

    def set_gl(self, val):
        """A callback function to handle the OpenCV slider to select the green lower bound"""
        self.gl = val

    def set_gh(self, val):
        """A callback function to handle the OpenCV slider to select the green upper bound"""
        self.gh = val

    def set_bl(self, val):
        """A callback function to handle the OpenCV slider to select the blue lower bound"""
        self.bl = val

    def set_bh(self, val):
        """A callback function to handle the OpenCV slider to select the blue upper bound"""
        self.bh = val


def appx_best_fit_ngon(mask_cv2, n: int = 4) -> list[(int, int)]:
    # convex hull of the input mask
    mask_cv2_gray = cv2.cvtColor(mask_cv2, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(
        mask_cv2_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    hull = cv2.convexHull(contours[0])
    hull = np.array(hull).reshape((len(hull), 2))

    # to sympy land
    hull = [sympy.Point(*pt) for pt in hull]

    # run until we cut down to n vertices
    while len(hull) > n:
        best_candidate = None

        # for all edges in hull ( <edge_idx_1>, <edge_idx_2> ) ->
        for edge_idx_1 in range(len(hull)):
            edge_idx_2 = (edge_idx_1 + 1) % len(hull)

            adj_idx_1 = (edge_idx_1 - 1) % len(hull)
            adj_idx_2 = (edge_idx_1 + 2) % len(hull)

            edge_pt_1 = sympy.Point(*hull[edge_idx_1])
            edge_pt_2 = sympy.Point(*hull[edge_idx_2])
            adj_pt_1 = sympy.Point(*hull[adj_idx_1])
            adj_pt_2 = sympy.Point(*hull[adj_idx_2])

            subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
            angle1 = subpoly.angles[edge_pt_1]
            angle2 = subpoly.angles[edge_pt_2]

            # we need to first make sure that the sum of the interior angles the edge
            # makes with the two adjacent edges is more than 180°
            if sympy.N(angle1 + angle2) <= sympy.pi:
                continue

            # find the new vertex if we delete this edge
            adj_edge_1 = sympy.Line(adj_pt_1, edge_pt_1)
            adj_edge_2 = sympy.Line(edge_pt_2, adj_pt_2)
            intersect = adj_edge_1.intersection(adj_edge_2)[0]

            # the area of the triangle we'll be adding
            area = sympy.N(sympy.Triangle(
                edge_pt_1, intersect, edge_pt_2).area)
            # should be the lowest
            if best_candidate and best_candidate[1] < area:
                continue

            # delete the edge and add the intersection of adjacent edges to the hull
            better_hull = list(hull)
            better_hull[edge_idx_1] = intersect
            del better_hull[edge_idx_2]
            best_candidate = (better_hull, area)

        if not best_candidate:
            raise ValueError("Could not find the best fit n-gon!")

        hull = best_candidate[0]

    # back to python land
    hull = [(int(x), int(y)) for x, y in hull]

    return hull


def move_closest_point_towards(target_point, points, step=1.0):
    """Move the target_point a little closer to the nearest point in `points`.

    Parameters
    - target_point: (x, y) tuple or array-like coordinates of the point to move.
    - points: sequence or ndarray of shape (N,2) containing (x,y) points.
    - step: positive float distance to move the target toward the nearest point.

    Returns
    - new_target: ndarray shape (2,) with the moved target coordinates.
    - idx: index of the closest point in `points`.

    Notes
    - If the closest point equals the target, no movement is performed.
    - If `step` is larger than the distance to the closest point, the
      target will be moved exactly onto that point.
    """
    arr = np.asarray(points)
    if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 2:
        pts = arr.reshape((-1, 2)).astype(float)
    elif arr.ndim == 2 and arr.shape[1] == 2:
        pts = arr.astype(float)
    elif arr.ndim == 1 and arr.dtype == object:
        # list of tuples
        pts = np.asarray([tuple(p) for p in arr], dtype=float)
    else:
        raise ValueError("points must be a sequence of (x,y) pairs")

    tgt = np.asarray(target_point, dtype=float).reshape(2,)

    # compute vector from target to each candidate point
    deltas = pts - tgt  # shape (N,2): vector from target -> candidate
    dists = np.linalg.norm(deltas, axis=1)
    idx = int(np.argmin(dists))
    dist = dists[idx]

    if dist == 0 or step == 0:
        return tgt, idx

    move = min(step, dist)
    direction = deltas[idx] / dist
    new_tgt = tgt + direction * move

    return new_tgt, idx


if __name__ == "__main__":
    test = testing()
    test.loop_wrapper()
