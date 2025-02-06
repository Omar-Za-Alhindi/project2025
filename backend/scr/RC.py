import cv2
from OCR import Processing as pr
import numpy as np
import re
import math

# input test image output Coordinates list of the test image
# input path output list of column
def get_Column(image):
    c = []
    img = cv2.imread(image)
    dilation = pr.pro1(img)
    _, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im2 = img.copy()
    prev = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h>7:

            c.append((x, y, w, h))
            # cv2.rectangle(img, (x,y) ,(x+w,y+h),(0,255,0),1)
            # cv2.imshow('ss',img)
            # cv2.waitKey(0)
        prev = y

    c.sort()
    return c

# get the Coordinates of rows
# input path  output list of rows
def get_Rows(image):
    img1 = cv2.imread(image)
    img1 = pr.resize(img1, 300, 300)
    dilation = pr.pro1(img1)
    # cv2.imshow('dilation' ,dilation)
    d = dilation.copy()
    _, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im2 = img1.copy()
    i = 0
    first = contours[0]
    prev = 0
    prevx = 0
    rows = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if y <= 290 and x and y != 0:
            if not rows:
                rows.append(y)
                # cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if pr.RowThere(rows, y):
                print('row there')
            else:
                rows.append(y)
                # cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.rectangle(d, (x, y), (x + w, y + h), (0, 255, 0), 2)
        i = i + 1
        # cv2.imshow('im2' ,im2)
        # cv2.imshow('d' ,d)

    # rows.sort()
    # print(rows)
    # print(len(rows))
    new_rows = []
    sc = cv2.imread('OCR/new_image.jpg')
    for i in rows:
        i = i*len(sc)/len(img1)
        new_rows.append(int(i))
    new_rows.sort()

    for i in range(len(new_rows)):
        pr.CleanRows(new_rows,len(sc))
    new_rows.append(new_rows[-1] + 25)
    # rows[0] = 0

    # print(rows)
    # print(new_rows)
    # print(len(new_rows))
    return new_rows

# matching for first row
def Tmatching(img, t):
    List = []
    img_rgb = cv2.imread(img)
    img_rgb = cv2.resize(img_rgb, (800, 800), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    if t == 'K':
        template = cv2.imread('OCR/Template Matching/Katrangi_Test.jpg', 0)
    elif t == 'S':
        template = cv2.imread('OCR/Template Matching/Shami_Test.jpg', 0)

    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.4
    loc = np.where(res >= threshold)
    print(loc[0][0])
    if not loc[0][0]:
        print('dfdfdfdf')
        return
    roi = img_rgb[loc[0][0]:loc[0][0] + h, loc[1][0] + 3:loc[1][0] + w]
    crop_img = img_rgb[loc[0][0] - 5:img_rgb.shape[1], loc[1][0]:img_rgb.shape[0]]
    List.append((loc[1][0], loc[0][0], w, h))
    # for pt in zip(*loc[::-1]):
    #     # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
    #     # cv2.imshow('ssss',img_rgb)
    #     # cv2.waitKey(0)
    #
    #     roi = img_rgb[pt[1]:pt[1]+h , pt[0]+3:pt[0]+w ]
    #     # c = img_rgb[pt, (pt[0] + w, pt[1] + h)]
    #     crop_img = img_rgb[pt[1]-5:img_rgb.shape[1] , pt[0]:img_rgb.shape[0]]
    #     List.append((pt[0] , pt[1] , w , h))
    # cv2.imshow("c",c)
    # print(pt)
    # print(pt[0] + w, pt[1] + h)

    # Show the final image with the matched area.
    # cv2.imshow('Detected', img_rgb)
    # cv2.imshow("d",roi)
    # cv2.imshow("crop",crop_img)
    # print(List[1])
    cv2.imwrite('OCR/test.jpg', roi)
    cv2.imwrite('OCR/new_image.jpg', crop_img)
    new_img = get_Column('OCR/test.jpg')
    # print(new_img)
    new_x = pr.Avg(new_img[0][0], new_img[1][0])
    new_x = int(new_x)
    # print(new_x)
    # print(List[1])
    # x = List[1]
    # print(new_img[0][0] ,new_img[1][0])
    # print(x[0] , new_x , x[1] ,img_rgb.shape[1])
    new_crop = img_rgb[loc[0][0]:img_rgb.shape[1], loc[1][0]:new_x]
    # print(new_crop[0] ,new_crop[1])
    cv2.imwrite('OCR/rows_crop.jpg', new_crop)

