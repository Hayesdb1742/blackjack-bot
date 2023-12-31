import VideoStream
import cv2 as cv
import Cards
import time

font = cv.FONT_HERSHEY_SIMPLEX
frame_rate_calc = 1
freq = cv.getTickFrequency()
FRAME_RATE = 10

stream = VideoStream.VideoStream((1920, 1080), 1, FRAME_RATE, 0).start()
time.sleep(1)

cam_quit = 0

while cam_quit == 0:
    image = stream.read()
    if len(image) == 0:
        print("image array is empty")
        stream.stop()
    t1 = cv.getTickCount()
    
    cv.putText(image, "FPS: " + str(int(frame_rate_calc)), (10,26), font, 0.7, (255,0,255), 2, cv.LINE_AA)

    cv.imshow("Card Detector", image)

    t2 = cv.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1

    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1
cv.destroyAllWindows()
stream.stop()

