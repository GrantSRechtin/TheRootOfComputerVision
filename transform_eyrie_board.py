import cv2
import math
import numpy as np
from main import player_board


def transform_board(birdsong_corners, transformed_corners, image):
    """
    Given the bounding corners of birdsong caption on eyrie board,
    return the transformation to flatten out the board image.

    Args:
        birdsong_corners - A list of the four corners of the birdsong
            caption in the unmodified image, containing tuples of floats
            in the form (x, y).
        transformed_corners - A list of the four corners of the birdsong
            caption after transformation, containing tuples of floats
            in the form (x, y).
            Order should be top-left, bottom-left, top-right, bottom-right.
        image - The unmodified cv2 image.

    returns:
        A cv2 image that is flat and square.
    """
    sorted_corners = sorted(birdsong_corners, key=lambda item: item[0])
    most_left = sorted_corners[0]
    second_left = ()
    remaining_indices = ()
    # Arbitrarily large number
    prev_dist = 10000
    for i in enumerate(sorted_corners[1:]):
        new_dist = math.dist(most_left, i[1])
        if new_dist < prev_dist:
            prev_dist = new_dist
            second_left = i[1]
            remaining_indices = (1 + (i[0] + 1) % 3, 1 + (i[0] + 2) % 3)
    ordered_corners = sorted([most_left, second_left], key=lambda item: item[1])
    remaining_corners = [
        sorted_corners[remaining_indices[0]],
        sorted_corners[remaining_indices[1]],
    ]
    remaining_corners = sorted(remaining_corners, key=lambda item: item[1])
    ordered_corners += [remaining_corners[0], remaining_corners[1]]

    img_width = 700
    # Define the max height of the final image
    img_height = 550
    # Define input and output corners to generation transformation
    input_pts = np.float32(ordered_corners)
    output_pts = np.float32(transformed_corners)
    persp_trans = cv2.getPerspectiveTransform(input_pts, output_pts)
    # Apply transformation to image
    trans_image = cv2.warpPerspective(
        image, persp_trans, (img_width, img_height), flags=cv2.INTER_NEAREST
    )
    board_width = 273.7
    perc_uncertainty = 0.01
    # Width of image birdsong divided by the actual width
    scale_factor = abs(transformed_corners[0][0] - transformed_corners[2][0]) / 110
    x_max = round(
        transformed_corners[1][0]  # Bottom left corner
        + scale_factor * (board_width - 8.5)
        + perc_uncertainty * scale_factor * (board_width - 8.5)
    )
    x_min = round(
        transformed_corners[1][0]  # Bottom left corner
        - scale_factor * 8.5
        - perc_uncertainty * scale_factor * 8.5
    )

    return trans_image[:, x_min:x_max], scale_factor


# def click_event(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(f"Coordinates: ({x}, {y})")
#         font = cv2.FONT_HERSHEY_PLAIN
#         cv2.putText(
#             test_image.img, f"({x}, {y})", (x, y), font, 2.5, (255, 255, 0), 1, 10
#         )
#         cv2.imshow("image", test_image.img)


# test_image = test()

# while test_image.active:
#     cv2.namedWindow("Original image")
#     cv2.namedWindow("Transformed image")

#     # birdsong_corners = [(47, 241), (44, 263), (285, 257), (283, 281)]
#     # birdsong_corners = [(295, 541), (300, 578), (650, 482), (657, 518)]
#     birdsong_corners = [(1046, 3048), (1163, 3086), (1389, 2198), (1492, 2224)]
#     transformed_corners = [(46, 258), (46, 276), (204.4, 258), (204.4, 276)]
#     cv2.imshow("Original image", test_image.img)
#     trans_image = transform_board(birdsong_corners, transformed_corners, test_image.img)
#     cv2.imshow("Transformed image", trans_image)
#     cv2.setMouseCallback("Original image", click_event)

#     key = cv2.waitKey(20)
#     if key == 27:
#         cv2.destroyAllWindows()
#         test_image.vc.release()
#         test_image.active = False
