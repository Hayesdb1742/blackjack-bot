import cv2 as cv
import numpy as np

CARD_MAX_AREA = 120000
CARD_MIN_AREA = 50000

image = cv.imread('test_image.jpg')

imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
cnts, hier = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(image, cnts, -1, (0,255,0), 3)
index_sort = sorted(range(len(cnts)), key=lambda i : cv.contourArea(cnts[i]), reverse=True)

cnts_sort = []
hier_sort = []
cnt_is_card = np.zeros(len(cnts), dtype=int)

for i in index_sort:
    cnts_sort.append(cnts[i])
    hier_sort.append(hier[0][1])

for i in range(len(cnts_sort)):
    size = cv.contourArea(cnts_sort[i])
    peri = cv.arcLength(cnts_sort[i], True)
    approx = cv.approxPolyDP(cnts_sort[i], 0.01*peri, True)
    if ((size > CARD_MIN_AREA) and (len(approx) == 6)): 
        drawing  = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        cv.drawContours(drawing, [cnts_sort[i]], -1, (0,255,0), 2)
        print(len(approx))
        cv.imshow('Contours', drawing)
        cv.waitKey(0)

    
cv.destroyAllWindows()


# cv.imshow('Gray Image', image)
# cv.waitKey(0)
# cv.destroyAllWindows()