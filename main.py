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
    """Player board controller for Eyrie game.

    Loads and processes a board image.
    Locates the Birdsong region and computes perspective transform 
        so the board can be analyzed
    Detects roosts and cards on the oriented board and displays them

    Usage:
        Press SPACE to orient the board (first time) or to update detections
        Press ESC to quit.
    """

    def __init__(self):
        """Initializes the player board"""

        self.active = True
        self.cv_image = None
        self.current_image = None

        self.scale_factor = None  # returned following image transformation
        self.oriented = False  # Set to true following image transformation

        # Tested Images of Boards
        # path = "Images/IMG_1695.JPG"
        path = "Images/52B-RootBaseFactionBoardwithComponents-Editv2-web.webp"
        # path = "Images/board_with_table.png"

        self.img = cv2.imread(path)

        self.clear = copy.copy(self.img)

        self.img = hp.blur(self.img)

        # Initial Birdsong Calibration Color Range
        self.birdsong_init = (70, 155, 196, 82, 196, 240)

        # Birdsong Color Range
        self.birdsong = (70, 130, 150, 130, 200, 240)

        # Main
        self.binary_calibration = [70, 130, 150, 130, 200, 240]

        # if self.vc.isOpened():
        self.cv_image = self.img  # self.vc.read()
        self.current_image = copy.copy(self.cv_image)

    def loop_wrapper(self):
        """loops run_loop"""

        # color_testing.test(self.img)
        cv2.namedWindow("video_window", cv2.WINDOW_NORMAL)

        while self.active:
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):
        """Updates the current image and closes the system, orients the board, or prints the current state, depending on user input"""
        if not self.current_image is None:

            cv2.imshow("video_window", self.current_image)

            key = cv2.waitKey(20)
            if key == 27:  # Escape
                self.esc()
            elif key == 32:  # Space

                if not self.oriented:
                    self.board_orientation()
                else:
                    self.update_board()

    def update_board(self):
        """Prints the current roosts and cards in each slot on the player board"""
        
        contours = roost_detection_test.detect_roosts(self)

        print(f"Roosts: {len(contours)}")

        recruit, move, battle, build = decree_detection.detect_decree(
            self.current_image, self.scale_factor)

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
        """transforms the board to a top down view"""

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
        """finds the coordinates of the 4 corners of the birdsong region of the board"""
        xs = [c[0][0] for c in contour]
        ys = [c[0][1] for c in contour]
        b_width = abs(max(xs) - min(xs))
        b_height = abs(max(ys) - min(ys))

        w = len(self.cv_image)
        h = len(self.cv_image[0])

        # Finds the region to look for birdsong
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

        birdsong_region = cv2.bitwise_and(binary_image, focus_region)
        contours = hp.create_binary_contours(birdsong_region)

        blank_image = np.zeros((h, w, 3), np.uint8)
        cv2.drawContours(blank_image, [contours[0][1]], 0, (0, 0, 255), 1)

        # Finds quadrilateral encompassing all points in the contour
        app = hp.appx_best_fit_ngon(blank_image)
        pts = np.array(app, dtype=np.int32)

        # Shifts points to closest point within the contour
        for i in range(len(pts)):
            pts[i], d = hp.move_closest_point_towards(
                pts[i], contours[0][1], 2)
            while d > 3:
                pts[i], d = hp.move_closest_point_towards(
                    pts[i], contours[0][1], 2)

        # icontour = self.cv_image.copy()
        # cv2.drawContours(icontour, [contour], 0, (0, 0, 255), 1)

        # result = self.cv_image.copy()
        # cv2.polylines(result, [pts], True, (0, 0, 255), 2)

        # cv2.namedWindow("CONTOUR", cv2.WINDOW_NORMAL)
        # cv2.imshow("CONTOUR", icontour)

        # cv2.namedWindow("QUAD", cv2.WINDOW_NORMAL)
        # cv2.imshow("QUAD", result)

        return pts

    def display_cards(self, section):
        """Helps print the cards in a pretty way"""
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
        """Closes the system"""
        cv2.destroyAllWindows()
        self.vc.release()
        self.active = False

if __name__ == "__main__":
    playerBoard = player_board()
    playerBoard.loop_wrapper()
