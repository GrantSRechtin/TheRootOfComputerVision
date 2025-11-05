import cv2
import numpy as np
from initial_test import testing as test


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
        image - The unmodified cv2 image.

    returns:
        A cv2 image that is flat and square.
    """
    # Define the max width of the final image
    img_width = 700
    # Define the max height of the final image
    img_height = 550
    # Define input and output corners to generation transformation
    input_pts = np.float32(birdsong_corners)
    output_pts = np.float32(transformed_corners)
    persp_trans = cv2.getPerspectiveTransform(input_pts, output_pts)
    # Apply transformation to image
    trans_image = cv2.warpPerspective(
        image, persp_trans, (img_width, img_height), flags=cv2.INTER_NEAREST
    )
    board_width = 273.7
    perc_uncertainty = 0.01
    scale_factor = abs(transformed_corners[0][0] - transformed_corners[2][0]) / 110
    x_max = round(
        transformed_corners[0][0]
        + scale_factor * (board_width - 8.5)
        + perc_uncertainty * scale_factor * (board_width - 8.5)
    )
    x_min = round(
        transformed_corners[0][0]
        - scale_factor * 8.5
        - perc_uncertainty * scale_factor * 8.5
    )

    return trans_image#[:, x_min:x_max]


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
