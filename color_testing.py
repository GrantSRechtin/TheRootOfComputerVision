import cv2
import time

binary_calibration = [0,0,0,255,255,255]

def set_rl(val):
    """A callback function to handle the OpenCV slider to select the red lower bound"""
    binary_calibration[0] = val


def set_rh(val):
    """A callback function to handle the OpenCV slider to select the red upper bound"""
    binary_calibration[3] = val


def set_gl(val):
    """A callback function to handle the OpenCV slider to select the green lower bound"""
    binary_calibration[1] = val


def set_gh(val):
    """A callback function to handle the OpenCV slider to select the green upper bound"""
    binary_calibration[4] = val


def set_bl(val):
    """A callback function to handle the OpenCV slider to select the blue lower bound"""
    binary_calibration[2] = val


def set_bh(val):
    """A callback function to handle the OpenCV slider to select the blue upper bound"""
    binary_calibration[5] = val


def test(cv_image, init_colors=[0, 0, 0, 255, 255, 255]):

    # update the module-level list in-place so the slider callbacks
    # (which modify the module list) and this function share the same
    # object. Assigning to the name `binary_calibration` here would
    # create a local variable and shadow the module list.
    binary_calibration[:] = init_colors

    cv2.namedWindow("binary_window", cv2.WINDOW_NORMAL)
    cv2.createTrackbar(
        "red lower bound", "binary_window", binary_calibration[0], 255, set_rl
    )
    cv2.createTrackbar(
        "red upper bound", "binary_window", binary_calibration[3], 255, set_rh
    )
    cv2.createTrackbar(
        "green lower bound", "binary_window", binary_calibration[1], 255, set_gl
    )
    cv2.createTrackbar(
        "green upper bound", "binary_window", binary_calibration[4], 255, set_gh
    )
    cv2.createTrackbar(
        "blue lower bound", "binary_window", binary_calibration[2], 255, set_bl
    )
    cv2.createTrackbar(
        "blue upper bound", "binary_window", binary_calibration[5], 255, set_bh
    )

    while True:
        time.sleep(0.1)

        print(binary_calibration)

        binary_image = cv2.inRange(
            cv_image, (binary_calibration[0], binary_calibration[1], binary_calibration[2]), (binary_calibration[3], binary_calibration[4], binary_calibration[5]))
        cv2.imshow('binary_window', binary_image)

        key = cv2.waitKey(20)
        if key == 27:  # Escape
            cv2.destroyAllWindows()
            import sys
            sys.exit()
