import cv2
import numpy as np
from initial_test import testing


def detect_roosts(testing_instance):
    """
    Given an image of the roosts section of the eyrie player board,
    find roost tokens.

    args:
        testing_instance - An instance of the class "testing".

    returns:
        A list of cv2 contours corresponding to each roost.
    """
    # Define bounds for Red, Green, and Blue to include in binary image
    rl = 120
    rh = 190
    gl = 50
    gh = 90
    bl = 25
    bh = 290
    # Generate binary image
    binary_image = cv2.inRange(testing_instance.img, (rl, gl, bl), (rh, gh, bh))
    # Uncomment to display binary image
    # cv2.namedWindow("Binary image")
    # cv2.imshow("Binary image", binary_image)

    # Generate contours
    contours = testing_instance.create_binary_contours(binary_image)
    # Filter contours by area
    roost_contour = []
    for i in contours:
        if 1250 > i[0] > 1000:
            roost_contour.append(i[1])
    return roost_contour


if __name__ == "__main__":
    test_image = testing()
    while test_image.active:
        roost_contour = detect_roosts(test_image)
        cv2.namedWindow("Original image")
        cv2.imshow("Original image", test_image.img)
        for i in enumerate(roost_contour):
            cv2.drawContours(
                test_image.img,
                roost_contour,
                i[0],
                (0, 0, 255),
                1,
            )

        key = cv2.waitKey(20)
        if key == 27:
            cv2.destroyAllWindows()
            test_image.vc.release()
            test_image.active = False
