import VideoStream
import cv2 as cv
import Cards
import os
import time

font = cv.FONT_HERSHEY_SIMPLEX
frame_rate_calc = 1
freq = cv.getTickFrequency()
FRAME_RATE = 10

stream = VideoStream.VideoStream((1920, 1080), 1, FRAME_RATE, 0).start()
time.sleep(1)

path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks(path + '/Card_Imgs/')
train_suits = Cards.load_suits(path + '/Card_Imgs/')
cam_quit = 0

while cam_quit == 0:
    image = stream.read()
    if len(image) == 0:
        print("image array is empty")
        stream.stop()
    t1 = cv.getTickCount()
    
    pre_process = Cards.preprocess_image(image)

    cnts_sort, cnt_is_card = Cards.find_cards(pre_process)

    if len(cnts_sort):
        cards = []
        k = 0
        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):
                cards.append(Cards.preprocess_card(cnts_sort[i], image))
                cards[k].best_rank_match,cards[k].best_suit_match, cards[k].rank_diff, cards[k].suit_diff = Cards.match_card(cards[k], train_ranks, train_suits)
                #cv.imwrite('testImages/rank.jpg', cards[k].rank_img)
                # cv.imwrite('testImages/card_suit.jpg', cards[k].suit_img)
                image = Cards.draw_results(image, cards[k])
                k += 1
        
        if len(cards) != 0:
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv.drawContours(image, temp_cnts, -1, (255,0,0), 2)

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
