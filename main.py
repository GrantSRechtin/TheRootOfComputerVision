import cv2
import time
import numpy as np
import sympy
import copy
import transform_eyrie_board
import roost_detection_test
import decree_detection
import color_testing
import helper_functions as hp
from operator import itemgetter


# Adjust range if no area found


class player_board:

    def __init__(self):

        self.active = True
        self.cv_image = None
        self.bt = False

        self.scale_factor = None

        self.current_image = None

        self.oriented = False

        self.i = 0

        self.vc = cv2.VideoCapture(0)  # Webcam w=640 h=480

        path_base = "board_with_table.png"
        self.base_img = cv2.imread(path_base)
        self.sw = len(self.base_img)
        self.sh = len(self.base_img[0])

        #path = "IMG_1695.JPG"
        path = "52B-RootBaseFactionBoardwithComponents-Editv2-web.webp"
        #path = "board_with_table.png"
        self.img = cv2.imread(path)

        self.clear = copy.copy(self.img)

        self.img = hp.blur(self.img)

        # Birdsong
        self.birdsong = (70, 130, 150, 130, 200, 240)

        # Birdsong Init
        self.birdsong_init = (0, 155, 196, 82, 196, 240)

        # Main
        self.binary_calibration = [70, 130, 150, 130, 200, 240]

        #if self.vc.isOpened():
        self.cv_image = self.img  # self.vc.read()
        self.current_image = copy.copy(self.cv_image)

    def loop_wrapper(self):
        """loops run_loop"""

        #color_testing.test(self.img)
        cv2.namedWindow("video_window", cv2.WINDOW_NORMAL)

        while self.active:
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):
        if not self.current_image is None:

            cv2.imshow("video_window", self.current_image)

            key = cv2.waitKey(20)
            if key == 27:  # Escape
                self.esc()
            elif key == 32: # Space

                if not self.oriented:
                    self.board_orientation()
                else:
                    self.update_board()

    def update_board(self):
        contours = roost_detection_test.detect_roosts(self)

        print(f"Roosts: {len(contours)}")

        recruit, move, battle, build = decree_detection.detect_decree(self.current_image, self.scale_factor)

        print("Recruit:")
        self.display_cards(recruit)
        print("Move:")
        self.display_cards(move)
        print("Battle:")
        self.display_cards(battle)
        print("Build:")
        self.display_cards(build)

        for i in enumerate(contours):
            cv2.drawContours(
                self.current_image,
                contours,
                i[0],
                (0, 0, 255),
                1,
            )
    
    def board_orientation(self):

        bs_binary = cv2.inRange(
            self.cv_image, self.birdsong_init[0:3], self.birdsong_init[3:6]
        )

        contours = hp.create_binary_contours(bs_binary)
        pts = self.find_birdsong_border(contours[0][1], self.birdsong)

        transformed_corners = [
            (46, 258), (46, 276), (204.4, 258), (204.4, 276)]
        self.translated_image, self.scale_factor = transform_eyrie_board.transform_board(
            pts, transformed_corners, self.cv_image
        )

        # cv2.namedWindow("Original image")
        # cv2.imshow("Original image", self.cv_image)

        self.current_image = copy.copy(self.translated_image)

        self.oriented = True

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

        focus_region = np.zeros((w, h), dtype=np.uint8)
        cv2.rectangle(
            focus_region,
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

        binary_adjustment_image = cv2.inRange(
            self.clear,
            (c_range[0], c_range[1], c_range[2]),
            (c_range[3], c_range[4], c_range[5]),
        )

        birdsong_region = cv2.bitwise_and(binary_image, focus_region)
        birdsong_adjustment_image = cv2.bitwise_and(binary_adjustment_image, focus_region)
        contours = hp.create_binary_contours(birdsong_region)

        birdsong_adjustment_points = hp.map_to_list(birdsong_adjustment_image)

        blank_image = np.zeros((h, w, 3), np.uint8)
        cv2.drawContours(blank_image, [contours[0][1]], 0, (0, 0, 255), 1)

        app = hp.appx_best_fit_ngon(blank_image)
        pts = np.array(app, dtype=np.int32)

        for i in range(len(pts)):
            pts[i], d = hp.move_closest_point_towards(pts[i], contours[0][1],2)
            while d > 3:
                pts[i], d = hp.move_closest_point_towards(pts[i], contours[0][1],2)

        contour = self.cv_image.copy()
        cv2.drawContours(contour, [contours[0][1]], 0, (0, 0, 255), 1)

        result = self.cv_image.copy()
        cv2.polylines(result, [pts], True, (0, 0, 255), 2)

        # cv2.namedWindow("CONTOUR", cv2.WINDOW_NORMAL)
        # cv2.imshow("CONTOUR", contour)

        # cv2.namedWindow("QUAD", cv2.WINDOW_NORMAL)
        # cv2.imshow("QUAD", result)

        return pts

    def display_cards(self, section):
        print("    ", end=" ")
        if section[0] > 0:
            print(f"Mice: {section[0]} ", end=" ")
        if section[1] > 0:
            print(f"Bunnies: {section[1]} ", end=" ")
        if section[2] > 0:
            print(f"Foxes: {section[2]} ", end=" ")
        if section[3] > 0:
            print(f"Birds: {section[3]} ", end=" ")
        print("")

    def esc(self):
        cv2.destroyAllWindows()
        self.vc.release()
        self.active = False

    def set_border(self, xs, ys):
        print(xs[0], xs[1])
        print(ys[1], ys[0])
        self.cv_image = self.cv_image[ys[1]: ys[0], xs[0]: xs[1]]
        cv2.createTrackbar(
            "red lower bound", "binary_window", self.binary_calibration[0], 255, self.set_rl
        )
        cv2.createTrackbar(
            "red upper bound", "binary_window", self.binary_calibration[3], 255, self.set_rh
        )
        cv2.createTrackbar(
            "green lower bound", "binary_window", self.binary_calibration[1], 255, self.set_gl
        )
        cv2.createTrackbar(
            "green upper bound", "binary_window", self.binary_calibration[4], 255, self.set_gh
        )
        cv2.createTrackbar(
            "blue lower bound", "binary_window", self.binary_calibration[2], 255, self.set_bl
        )
        cv2.createTrackbar(
            "blue upper bound", "binary_window", self.binary_calibration[5], 255, self.set_bh
        )

if __name__ == "__main__":
    playerBoard = player_board()
    playerBoard.loop_wrapper()
