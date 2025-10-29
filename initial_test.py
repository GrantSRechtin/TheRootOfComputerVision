import cv2
import time
import numpy as np

class testing():

    def __init__(self):

        self.active = True
        self.cv_image = None

        self.vc = cv2.VideoCapture(0) # Webcam w=640 h=480
        
        self.img = cv2.imread("Eyrie_Dynasties_-_Faction_Board.webp")

        # Lighter Blue Background
        # self.rl = 29
        # self.rh = 98
        # self.bl = 58
        # self.bh = 192
        # self.gl = 31
        # self.gh = 115

        self.rl = 29
        self.rh = 98
        self.bl = 58
        self.bh = 192
        self.gl = 31
        self.gh = 115

        if self.vc.isOpened(): # try to get the first frame
            self.cv_image = self.img #self.vc.read()
        
    def loop_wrapper(self):
        """ loops run_loop """
        
        cv2.namedWindow('video_window')
        cv2.namedWindow('binary_window')
        cv2.namedWindow('image_info')
        cv2.createTrackbar('red lower bound', 'binary_window', self.rl, 255, self.set_rl)
        cv2.createTrackbar('red upper bound', 'binary_window', self.rh, 255, self.set_rh)
        cv2.createTrackbar('green lower bound', 'binary_window', self.gl, 255, self.set_gl)
        cv2.createTrackbar('green upper bound', 'binary_window', self.gh, 255, self.set_gh)
        cv2.createTrackbar('blue lower bound', 'binary_window', self.bl, 255, self.set_bl)
        cv2.createTrackbar('blue upper bound', 'binary_window', self.bh, 255, self.set_bh)

        while self.active:
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):

        if not self.cv_image is None:

            self.binary_image = cv2.inRange(self.cv_image, (self.bl,self.gl,self.rl), (self.bh,self.gh,self.rh))
            cv2.imshow('video_window', self.cv_image)
            cv2.imshow('binary_window', self.binary_image)

            edged = cv2.Canny(self.binary_image, 30, 200)
            cv2.imshow('Canny Edges After Contouring', edged)

            # contours, hierarchy = cv2.findContours(edged,
            #           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # cv2.drawContours(self.cv_image, contours, -1, (0, 255, 0), 3)
            #     cv2.imshow('Contours', self.cv_image)
            
            key = cv2.waitKey(20)
            if key == 27:
                cv2.destroyAllWindows()
                self.vc.release()
                self.active = False

            cv2.waitKey(5)

            print("Number of Contours Found = " + str(len(contours)))

    def set_rl(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """
        self.rl = val

    def set_rh(self, val):
        """ A callback function to handle the OpenCV slider to select the red upper bound """
        self.rh = val

    def set_gl(self, val):
        """ A callback function to handle the OpenCV slider to select the green lower bound """
        self.gl = val

    def set_gh(self, val):
        """ A callback function to handle the OpenCV slider to select the green upper bound """
        self.gh = val

    def set_bl(self, val):
        """ A callback function to handle the OpenCV slider to select the blue lower bound """
        self.bl = val

    def set_bh(self, val):
        """ A callback function to handle the OpenCV slider to select the blue upper bound """
        self.bh = val

test = testing()
test.loop_wrapper()
