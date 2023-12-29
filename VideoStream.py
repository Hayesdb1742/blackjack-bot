import cv2
from threading import Thread

class VideoStream:
    def __init__(self, resolution=(1920, 1080), phoneOrComputer=1, fps=30, src=0):


        self.phoneOrComputer = phoneOrComputer
        self.cam = None
        self.stopped = False
        self.frame = []
    def getCamera(self):
        self.cam = cv2.VideoCapture(0)


    def start(self):
        #threading
        Thread(target=self.update, name="updateThread").start()
        return self
    
    def update(self):
        while True:
            _, self.frame = self.cam.read()
            if self.stopped:
                return self.cam.release()

    def read(self):
        return self.frame
    def stop(self):
        self.stopped = True

    def showFeed(self):
        while(True):
            ret, frame = self.cam.read()
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        self.cam.release()
        cv2.destroyAllWindows()



