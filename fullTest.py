import VideoStream
import cv2 as cv
import Cards
import os
import time


path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks(path + '/Card_Imgs/')
train_suits = Cards.load_suits(path + '/Card_Imgs/')
image_quit = 0

while image_quit == 0:
    image = cv.imread("test/test_image.jpg")
    if len(image) == 0:
        print("image array is empty")
    
    pre_process = Cards.preprocess_image(image)

    cnts_sort, cnt_is_card = Cards.find_cards(pre_process)

    if len(cnts_sort):
        print("card detected")
        cards = []
        k = 0
        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):
                cards.append(Cards.preprocess_card(cnts_sort[i], image))
                cards[k].best_rank_match,cards[k].best_suit_match, cards[k].rank_diff, cards[k].suit_diff = Cards.match_card(cards[k], train_ranks, train_suits)
                image = Cards.draw_results(image, cards[k])
                k += 1
        
        if len(cards) != 0:
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv.drawContours(image, temp_cnts, -1, (255,0,0), 2)
    cv.imshow("Card Detector", image)
    cv.waitKey(0)
    image_quit=1
    
    
cv.destroyAllWindows()

