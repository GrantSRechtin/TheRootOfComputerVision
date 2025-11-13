import cv2
import numpy as np


def detect_decree(image, scale_factor):
    """
    Given an instance of the Eyrie player_board, detect each card
    and sort them by region.

    args:
        image - A cv2 image of an Eyrie Dynasties player board
        scale_factor - Float representing the ratio of pixels/mm

    returns:
        A list of four lists with strings indicating a card type. The
        four lists corresponds to the four regions on the board:
        recruit, move, battle, build.
    """
    recruit = []
    move = []
    battle = []
    build = []
    full_decree = [recruit, move, battle, build]
    # Space allocated for each category on board (mm)
    category_threshold = round(70 * scale_factor)
    board_width = 273.7
    # Approximate number of pixels in one card banner
    banner_area = [400000, 400000, 400000, 400000]
    # Half the width of the board in pixels
    board_center = round(scale_factor * board_width / 2)
    mouse_filter = [110, 280, 140, 190, 210, 250]
    bunny_filter = [50, 150, 200, 230, 200, 230]
    fox_filter = [50, 130, 45, 70, 80, 230]
    bird_filter = [190, 220, 170, 200, 110, 170]
    all_filters = [mouse_filter, bunny_filter, fox_filter, bird_filter]
    # cv2.namedWindow("Binary image")
    for filter in all_filters:
        # Define bounds for Red, Green, and Blue to include in binary image
        rl = filter[0]
        rh = filter[1]
        gl = filter[2]
        gh = filter[3]
        bl = filter[4]
        bh = filter[5]
        # Generate binary image
        binary_image = cv2.inRange(image, (rl, gl, bl), (rh, gh, bh))
        # Segment the image into four slices for each card location
        segment_one = binary_image[:, : (board_center - category_threshold)]
        segment_two = binary_image[
            :, (board_center - category_threshold) : board_center
        ]
        segment_three = binary_image[
            :, board_center : (board_center + category_threshold)
        ]
        segment_four = binary_image[:, (board_center + category_threshold) :]
        all_segments = [segment_one, segment_two, segment_three, segment_four]
        # Calculate the number of cards in each slice
        for i in range(4):
            full_decree[i] += [round(np.sum(all_segments[i]) / banner_area[i])]
        # exit_status = False
        # while exit_status is False:
        #     cv2.imshow("Binary image", binary_image)
        #     key = cv2.waitKey(20)
        #     if key == 27:
        #         exit_status = True
    return full_decree


if __name__ == "__main__":
    image = cv2.imread("board_with_table.png")
    detect_decree(image, 2.35)
