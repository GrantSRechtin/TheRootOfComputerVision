import cv2
import time
import numpy as np
import sympy
import copy
import transform_eyrie_board
import roost_detection_test
import decree_detection
import color_testing
from operator import itemgetter

def blur(img):
    h, w, channels = img.shape
    return cv2.resize(
        cv2.resize(img, (700, 550), interpolation=cv2.INTER_AREA), (w, h)
    )

def create_binary_contours(binary_image, invert=False):

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

def create_contours(image):

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

def map_to_list(map):
    coords = []
    h = len(map)
    w = len(map[0]) if h > 0 else 0
    for x in range(h):
        for y in range(w):
            if map[x][y] == 255:
                coords.append([x, y])
    return np.array(coords, dtype=int)

def move_closest_point_towards(target_point, points, step=1):
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
        #print(points)
        raise ValueError("points must be a sequence of (x,y) pairs")

    tgt = np.asarray(target_point, dtype=float).reshape(
        2,
    )

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

    return new_tgt, dist