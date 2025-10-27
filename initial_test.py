import cv2
import time

class testing():

    def __init__(self):

        self.rval = False
        self.cv_image = None

        self.vc = cv2.VideoCapture(0) # Webcam w=640 h=480
        
        self.img = cv2.imread("Eyrie_Dynasties_-_Faction_Board.webp")
        #self.img = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)

        # Ligher Blue Background Bounds
        # rl 29   rh 98   gl 31   gh 115   bl 58   bh 192

        self.rl = 1
        self.rh = 255
        self.bl = 1
        self.bh = 255
        self.gl = 1
        self.gh = 255

        if self.vc.isOpened(): # try to get the first frame
            self.cv_image = self.img #self.vc.read()
        else:
            self.rval = False
        
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

        while True:
            self.run_loop()
            time.sleep(0.1)

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

    def run_loop(self):

        if not self.cv_image is None:

            #self.rval, self.cv_image = self.vc.read()
            key = cv2.waitKey(20)
            if key == 27:
                cv2.destroyAllWindows()
                self.vc.release()

            self.binary_image = cv2.inRange(self.cv_image, (self.bl,self.gl,self.rl), (self.bh,self.gh,self.rh))
            cv2.imshow('video_window', self.cv_image)
            cv2.imshow('binary_window', self.binary_image)
            cv2.waitKey(5)

            print(f"len 1: {len(self.binary_image)} len 2: {len(self.binary_image[0])}")

test = testing()
test.loop_wrapper()
