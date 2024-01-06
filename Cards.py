import cv2 as cv
import numpy as np

#Adaptive thresh
BKG_THRESH = 60
CARD_THRESH = 30

CARD_MAX_AREA = 4000000
CARD_MIN_AREA = 25000

CORNER_WIDTH = 32
CORNER_HEIGHT= 110

RANK_WIDTH = 70
RANK_HEIGHT = 125

SUIT_WIDTH = 70
SUIT_HEIGHT = 100

RANK_DIFF_MAX = 5000
SUIT_DIFF_MAX = 1000    

font = cv.FONT_HERSHEY_SIMPLEX

class CardInfo:
    def __init__(self):
        self.contour = []
        self.width, self.height= 0,0
        self.corner_pts = []
        self.warp = []
        self.rank_img = []
        self.suit_img = []
        self.best_rank_match = "Unknown"
        self.best_suit_match = "Unknown"
        self.rank_diff = 0
        self.suit_diff = 0


class Train_ranks:
    def __init__(self):
        self.img = []
        self.name = "Placeholder"
        self.values = {}
    def setValues(self, ranks):
        self.ranks = ranks

    def getValues(self):
        for k,v in self.values:
            print(k + ": " + v)


class Train_suits:
    def __init__(self):
        self.img = []
        self.name = "Placeholder"
        self.values = {}
    def setValues(self, ranks):
        self.ranks = ranks

def load_ranks(filepath):

    train_ranks = []
    i =0
    for Rank in ['Ace','Two','Three','Four','Five','Six','Seven',
                 'Eight','Nine','Ten','Jack','Queen','King']:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank

        filename = Rank +'.jpg'
        train_ranks[i].img = cv.imread(filepath+ filename, cv.IMREAD_GRAYSCALE)
        i = i+1

    return train_ranks
    

def load_suits(filepath):
    train_suits = []
    i = 0

    for Suit in ['Spades','Diamonds', 'Clubs', 'Hearts']:
        train_suits.append(Train_suits())
        train_suits[i].name = Suit
        filename = Suit + '.jpg'
        train_suits[i].img = cv.imread(filepath + filename, cv.IMREAD_GRAYSCALE)
        i=i+1
    return train_suits

def preprocess_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)

    imgW, imgH = np.shape(image)[:2]
    bkg_level = gray[int(imgH/100)][int(imgW/2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv.threshold(blur, thresh_level, 255, cv.THRESH_BINARY)
    return thresh


def find_cards(thresh_image):

    cnts, hier = cv.findContours(thresh_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i: cv.contourArea(cnts[i]), reverse=True)

    if len(cnts) == 0:
        return [], []
    
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts), dtype=int)

    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])
    
    for i in range(len(cnts_sort)):
        size = cv.contourArea(cnts_sort[i])
        peri = cv.arcLength(cnts_sort[i], True)
        approx = cv.approxPolyDP(cnts_sort[i], 0.01*peri, True)
        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
        and (hier_sort[i][3] == -1) and ((len(approx) == 6) or (len(approx) ==4))):
            cnt_is_card[i] = 1


        return cnts_sort, cnt_is_card


def preprocess_card(contour, image):

    qCard = CardInfo()

    qCard.contour = contour
    peri = cv.arcLength(contour,True)
    approx = cv.approxPolyDP(contour, 0.01*peri, True)
    pts = np.float32(approx)

    qCard.corner_pts = pts

    x,y,w,h = cv.boundingRect(contour)
    qCard.width, qCard.height = w, h

    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]
    
    qCard.warp = flattener(image, pts, w, h)
    cv.imwrite("testImages/card.jpg", qCard.warp)
    Qcorner = qCard.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    Qcorner_zoom = cv.resize(Qcorner, (0,0), fx=4, fy=4)
    cv.imwrite("testImages/corner.jpg", Qcorner_zoom)
    whiteLevel = Qcorner_zoom[15, int((CORNER_WIDTH*4)/2)]
    thresh_level = whiteLevel - CARD_THRESH

    if( thresh_level <= 0):
        thresh_level = 1

    retval, query_thresh = cv.threshold(Qcorner_zoom, thresh_level, 255, cv.THRESH_BINARY_INV)

    Qrank = query_thresh[50:175, 0:128]
    Qsuit = query_thresh[176:426, 0:128]

    #need to find contours of suit and rank
    Qrank_cnts, hier = cv.findContours(Qrank, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv.contourArea, reverse=True)

    if len(Qrank_cnts) != 0:
        x1, y1, w1, h1 = cv.boundingRect(Qrank_cnts[0])
        Qrank_roi = Qrank[y1:y1+h1, x1: x1+w1]
        Qrank_sized = cv.resize(Qrank_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
        qCard.rank_img = Qrank_sized
    
    Qsuit_cnts, hier = cv.findContours(Qsuit, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    Qsuit_cnts = sorted(Qsuit_cnts, key=cv.contourArea, reverse=True)

    if len(Qsuit_cnts) != 0:
        x2, y2, w2, h2 = cv.boundingRect(Qsuit_cnts[0])
        Qsuit_roi = Qsuit[y2:y2+h2, x2:x2+w2]
        Qsuit_sized = cv.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        qCard.suit_img = Qsuit_sized
    return qCard


def match_card(qCard, train_ranks, train_suits):
    best_rank_match_diff = 10000
    best_suit_match_diff = 10000
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"
    i =0
    rankDiffValues = {}
    suitDiffValues = {}
    if (len(qCard.rank_img) != 0) and (len(qCard.suit_img) != 0):
        for Trank in train_ranks:
            diff_img = cv.absdiff(qCard.rank_img, Trank.img)
            rank_diff = int(np.sum(diff_img)/255)
            print(f"{Trank.name}:{rank_diff}") 
            rankDiffValues[Trank.name] = rank_diff

            if rank_diff < best_rank_match_diff:
                best_rank_diff_img = diff_img
                best_rank_match_diff = rank_diff   
                best_rank_name = Trank.name
            
        for Tsuit in train_suits: 
            diff_img = cv.absdiff(qCard.suit_img, Tsuit.img)
            suit_diff = int(np.sum(diff_img)/255)
            print(f"{Tsuit.name}:{suit_diff}")
            suitDiffValues[Tsuit.name] = suit_diff
            if suit_diff < best_suit_match_diff:
                best_suit_diff_img = diff_img
                best_suit_match_diff = suit_diff
                best_suit_match_name = Tsuit.name
    if (best_rank_match_diff < RANK_DIFF_MAX):
        best_rank_match_name = best_rank_name
    if (best_suit_match_diff < SUIT_DIFF_MAX):
        best_suit_match_name = best_suit_match_name

    #train_ranks.setValues(rankDiffValues)
    #train_suits.setValues(suitDiffValues)
    return best_rank_match_name, best_suit_match_name, best_rank_match_diff, best_suit_match_diff


def draw_results(image, qCard):
    x = qCard.center[0]
    y = qCard.center[1]

    cv.circle(image, (x,y), 5, (255,0,0), -1)

    rank_name = qCard.best_rank_match
    suit_name = qCard.best_suit_match

    cv.putText(image, (rank_name+' of'), (x-60, y-10), font, 5, (0,0,0), 3, cv.LINE_AA)
    cv.putText(image, (suit_name+' of'), (x-60, y-150), font, 5, (50,200, 200), 2, cv.LINE_AA)

    return image

def flattener(image, pts, w, h):
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, b  ottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
        
    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv.getPerspectiveTransform(temp_rect,dst)
    warp = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv.cvtColor(warp,cv.COLOR_BGR2GRAY)

        

    return warp

