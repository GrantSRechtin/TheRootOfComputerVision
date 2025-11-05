import cv2
import time
import numpy as np
from operator import itemgetter


class testing():

    def __init__(self):

        self.active = True
        self.cv_image = None
        self.cv_image_with_contours = None

        self.i = 0

        self.vc = cv2.VideoCapture(0)  # Webcam w=640 h=480

        path = "board_with_table.png"
        self.img = cv2.imread(path)
        self.img_contours = cv2.imread(path)        

        # self.img = self.blur(self.img)
        # self.img_contours = self.blur(self.img_contours)

        # Card Slots
        # self.rl = 0
        # self.rh = 255
        # self.gl = 0
        # self.gh = 41
        # self.bl = 38
        # self.bh = 130

        # Birdsong
        # self.rl = 200
        # self.rh = 240
        # self.gl = 140
        # self.gh = 200
        # self.bl = 70
        # self.bh = 150

        # Birdsong 2
        self.rl = 196
        self.rh = 240
        self.gl = 155
        self.gh = 196
        self.bl = 0
        self.bh = 82

        # Main
        # self.rl = 190
        # self.rh = 240
        # self.gl = 178
        # self.gh = 255
        # self.bl = 168
        # self.bh = 255

        if self.vc.isOpened():  # try to get the first frame
            self.cv_image = self.img  # self.vc.read()
            self.cv_image_with_contours = self.img_contours

    def loop_wrapper(self):
        """ loops run_loop """

        cv2.namedWindow('video_window')
        cv2.namedWindow('binary_window')
        cv2.namedWindow('contour_window')
        cv2.createTrackbar('red lower bound', 'binary_window',
                           self.rl, 255, self.set_rl)
        cv2.createTrackbar('red upper bound', 'binary_window',
                           self.rh, 255, self.set_rh)
        cv2.createTrackbar('green lower bound',
                           'binary_window', self.gl, 255, self.set_gl)
        cv2.createTrackbar('green upper bound',
                           'binary_window', self.gh, 255, self.set_gh)
        cv2.createTrackbar('blue lower bound', 'binary_window',
                           self.bl, 255, self.set_bl)
        cv2.createTrackbar('blue upper bound', 'binary_window',
                           self.bh, 255, self.set_bh)

        while self.active:
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):

        if not self.cv_image is None:

            
            self.binary_image = cv2.inRange(
                self.cv_image, (self.bl, self.gl, self.rl), (self.bh, self.gh, self.rh))

            #contours = self.create_contours()
            contours = self.create_binary_contours()

            cv2.imshow('video_window', self.cv_image)
            cv2.imshow('binary_window', self.binary_image)
            cv2.imshow('contour_window', self.cv_image_with_contours)

            key = cv2.waitKey(20)
            if key == 27:
                cv2.destroyAllWindows()
                self.vc.release()
                self.active = False
            elif key == 32:
                cv2.drawContours(
                    self.cv_image_with_contours, contours[self.i][1], -1, (0, 255, 0), 3)
                self.i += 1
            elif key == 9:
                xs, ys = self.find_border(contours[0][1])
                self.set_border(xs,ys)

            cv2.waitKey(5)
    
    def blur(self, img):
        h, w, channels = img.shape
        return cv2.resize(cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_AREA),(w,h))

    def create_binary_contours(self):

        edged = cv2.Canny(self.binary_image, 100, 200)
        contour_list, hierarchy = cv2.findContours(edged,
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour_areas = [cv2.contourArea(c) for c in contour_list]
        contours = [(contour_areas[i], contour_list[i])
                    for i in range(len(contour_list))]
        contours = sorted(contours, key=itemgetter(0))[::-1]

        return contours
    
    def create_contours(self):

        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 30, 200)
        contour_list, hierarchy = cv2.findContours(edged,
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour_areas = [cv2.contourArea(c) for c in contour_list]
        contours = [(contour_areas[i], contour_list[i])
                    for i in range(len(contour_list))]
        contours = sorted(contours, key=itemgetter(0))[::-1]

        return contours
    
    def find_border(self,contour):
        xl_ratio = 15/270
        xr_ratio = 345/270
        yb_ratio = 30/265
        yt_ratio = 180/265

        xs = [c[0][0] for c in contour]
        ys = [c[0][1] for c in contour]

        c_xs = (min(xs),max(xs))
        c_ys = (max(ys),min(ys))

        c_width = abs(max(xs) - min(xs))
        c_height = abs(max(ys) - min(ys))

        b_x = (int(-xl_ratio*c_width+c_xs[0]), int(xr_ratio*c_width+c_xs[1]))
        b_y = (int(yb_ratio*c_height+c_ys[0]), max(0,int(-yt_ratio*c_height+c_ys[1])))

        return b_x, b_y
    
    def set_border(self, xs, ys):
        print(xs[0],xs[1])
        print(ys[1],ys[0])
        self.cv_image = self.cv_image[ys[1]:ys[0],xs[0]:xs[1]]
        self.cv_image_with_contours = self.cv_image_with_contours[ys[1]:ys[0],xs[0]:xs[1]]

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




# ratios
# x 30 - 45 -- 315 - 660
# y 580 - 550 -- 285 - 100