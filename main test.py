import os
import glob
import cv2

import re
# importing executeString to run sql commands
import re
# import the fuzzywuzzy module
from fuzzywuzzy import fuzz
import cv2
import math
import numpy as np
from PIL import Image
import pytesseract
import cv2
from OCR import Processing as pr
import numpy as np
import re
import math
import cv2
from OCR import RC, Processing as pr
from OCR import TemplateMatching

column_number = -1
template = ''

def main(image):
    global column_number
    global template
    # image = "D:\\5\\Project\\Resize\\Katrangi\\NewResults\\newTest2.jpg"
    # image = "D:\\5\\Project\\Resize\\Katrangi\\NewResults\\New folder\\KatrangiTest12.jpg"
    template = TemplateMatching.tm(image)
    print('Template :', template)
    RC.Tmatching(image, template)
    img = cv2.imread('OCR/new_image.jpg')
    c = RC.get_Column('OCR/test.jpg')
    column_number = len(c)
    if template == 'S':
        column_number+=1
    print(c)
    print('Column :', len(c))
    r = RC.get_Rows('OCR/rows_crop.jpg')
    # try:
    #     RC.Tmatching(image, template)
    #     img = cv2.imread('new_image.jpg')
    #     c = RC.get_Column('test.jpg')
    #     print(c)
    #     print('Column :', len(c))
    #     r = RC.get_Rows('rows_crop.jpg')
    # except:
    #     r = RC.NoMatching(image)
    #     img = cv2.imread('new_img.jpg')
    #     c = RC.get_Column('new_test.jpg')

    # print(c)
    # print(r)
    if template == 'K':
        x0 = 0
        x1 = c[1][0] + 5
        x2 = pr.Avg(c[1][0] + c[1][2], c[2][0]) + 10
        x3 = pr.Avg(c[2][0] + c[2][2], c[3][0]) - 10
        x4 = c[4][0]
        x5 = img.shape[1]
        ax = [int(x0), int(x1), int(x2), int(x3), int(x4), int(x5)]
        a = len(c) * len(r)
    if template == 'S':
        x0 = 0
        x1 = c[1][0]
        x2 = c[1][0] + c[1][2]
        x3 = c[2][0] - 10
        x4 = img.shape[1]
        ax = [x0, x1, x2, x3, x4]

        # r = np.array(r)
        # r = r*2.17
        # print(r)
        #
        # print(ax)
        # print(a)
    r1 = 0
    counter = 0
    for i in range(len(r) - 1):
        for j in range(len(ax) - 1):
            new_img = img.copy()
            new_img1 = img.copy()
            # cv2.rectangle(img, (ax[j], int(r[i])), (ax[j + 1], int(r[i + 1])), (0, 255, 0), 2)
            # cv2.rectangle(new_img ,(ax[i],int(r[j])),(ax[i+1],int(r[j+1])) ,(0,255,0),2)
            # cv2.rectangle(new_img1 ,(ax[j],int(r[i])),(ax[j+1],int(r[i+1])) ,(0,255,0),2)
            # cv2.imshow('ne',new_img1)
            # cv2.waitKey(100)
            # for data
            s_img = new_img[int(r[i]):int(r[i+1]),ax[j]:ax[j+1]]
            cv2.imwrite('OCR/images/' + str(counter)+'.jpg',s_img)
            counter = counter+1

    # cv2.imshow('f', img)
    # cv2.waitKey(0)




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


# resize image to x ,y shape
def resize(img , x , y):
    re = cv2.resize(img, (x, y), interpolation=cv2.INTER_AREA)
    return re

# filters for contours
def pro(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)
    add = cv2.add(value, topHat)
    subtract = cv2.subtract(add, blackHat)
    blur = cv2.GaussianBlur(subtract, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    return thresh

# filters for contours
def pro1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
    return dilation

# Crop left part
def cropLeft(img):
    height = img.shape[0]
    width = img.shape[1]

    crop_img = img[0:height, 0:int(width / 3)]
    return crop_img

def crop(img , pointTop, pointDown):
    x = 0
    y = 0
    cropi = img[pointDown+4:800 , 0:800]
    cv2.rectangle(img , (0 , pointTop-6) , (800,pointDown+4) , (0,0,255),2)
    # cv2.imshow("crop",cropi)
    # cp = img[pointDown - 6: pointTop + 4, 0:800]
    # cv2.imshow("cp",cp)
    # cv2.imshow("",img)
    # cv2.waitKey(0)

def dpi(image , x , y):
    img = Image.open(image)
    img.save(image, dpi=(x, y))

def Ifthere(x,list):
    for i in list:
        if math.fabs(i[1] - x)<=15:
            return True

x = 0
y = 0
def row(image , list ):
    x = list[0][0]
    y = list[0][1]
    for i in list:
        crop = image[x:i[0] , y:i[1]]
        x = i[0]
        y = i[1]
        # cv2.imshow("",crop)
        # cv2.waitKey(0)

def Avg(x1, x2):
    sum = x1 + x2
    avg = sum / 2
    return avg

def RowThere(l, y):
        for i in l:
            if math.fabs(i - y) <= 3:
                return True


def CleanRows(l,x):
    first = l[0]
    q4 = x/4
    x = x - q4
    for i in l[1:]:
        if math.fabs(first - i) >= 70 and i >x:
            index = l.index(i)
            l.pop(index)

        first = i

# A more optimized version of the Levenshtein distance function using an array of previously calculated distances
def opti_leven_distance(a, b):
    # Create an empty distance matrix with dimensions len(a)+1 x len(b)+1
    dists = [[0 for _ in range(len(b) + 1)] for _ in range(len(a) + 1)]

    # a's default distances are calculated by removing each character
    for i in range(1, len(a) + 1):
        dists[i][0] = i
    # b's default distances are calulated by adding each character
    for j in range(1, len(b) + 1):
        dists[0][j] = j

    # Find the remaining distances using previous distances
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            # Calculate the substitution cost
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1

            dists[i][j] = min(
                # Removing a character from a
                dists[i - 1][j] + 1,
                # Adding a character to b
                dists[i][j - 1] + 1,
                # Substituting a character from a to b
                dists[i - 1][j - 1] + cost
            )

    return dists[-1][-1]




# correct unit to be like the data ###X109/l(10^9) to 10e9/L
def unitCorrection(text):
    # result = []
    for line in text:
        line['unit'] = re.sub(r'(X10)(()*)(([0-9])+)/l', r'10e\4/L', line['unit'])
        # result.append(line)
        # line['unit'] = 'hello'
    # return result


# correct biomarkers in case of error in reading
def biomarkerFixer(text):
    createCheckerFile('biomarker')
    myChecker = SpellCheck('NLP/biomarker.txt')
    for line in text:
        line['test'] = re.sub(r'([a-z])(\()', r'\1 \2', line['test'])
        tempLine = ''
        for word in line['test'].split(' '):
            myChecker.check(word)
            minDis = 10000
            tempCorrect = ''
            for suggest in myChecker.suggestions():
                if suggest == word:
                    tempCorrect = suggest
                    break
                tempDis = opti_leven_distance(word, suggest)
                if tempDis < minDis and tempDis < len(word) / 3:
                    minDis = tempDis
                    tempCorrect = suggest
                    print(minDis, suggest, word)

            if tempCorrect != '':
                word = tempCorrect
            if tempLine != '':
                tempLine += ' '
            tempLine += word

        line['test'] = tempLine

def rangeNormlizer(text):
    for line in text:
        line['range'] = re.sub(r'([0-9])\s*-\s*([0-9])', r'\1-\2', line['range'])

def dropLastDatedResult(text):
    for line in text:
        line.pop('lastdatedresult')


# spellcheck main class
class SpellCheck:

    # initialization method
    def __init__(self, word_dict_file='words.txt'):
        # open the dictionary file
        self.file = open(word_dict_file, 'r')

        # load the file data in a variable
        data = self.file.read()

        # store all the words in a list
        data = data.split(",")

        # change all the words to lowercase
        data = [i.lower() for i in data]

        # remove all the duplicates in the list
        data = set(data)

        # store all the words into a class variable dictionary
        self.dictionary = list(data)

    # string setter method
    def check(self, string_to_check):
        # store the string to be checked in a class variable
        self.string_to_check = string_to_check

    # this method returns the possible suggestions of the correct words
    def suggestions(self):
        # store the words of the string to be checked in a list by using a split function
        string_words = self.string_to_check.split()

        # a list to store all the possible suggestions
        suggestions = []

        # loop over the number of words in the string to be checked
        for i in range(len(string_words)):

            # loop over words in the dictionary
            for name in self.dictionary:

                # if the fuzzywuzzy returns the matched value greater than 80
                if fuzz.ratio(string_words[i].lower(), name.lower()) >= 75:
                    # append the dict word to the suggestion list
                    suggestions.append(name)

        # return the suggestions list
        return suggestions

    # this method returns the corrected string of the given input
    def correct(self):
        # store the words of the string to be checked in a list by using a split function
        string_words = self.string_to_check.split()

        # loop over the number of words in the string to be checked
        for i in range(len(string_words)):

            # initiaze a maximum probability variable to 0
            max_percent = 0
            tempName = ''

            # loop over the words in the dictionary
            for name in self.dictionary:

                # calulcate the match probability
                percent = fuzz.ratio(string_words[i].lower(), name.lower())

                percent -= abs(len(name) - len(string_words[i]))

                # if the fuzzywuzzy returns the matched value greater than 80
                if percent >= 83:

                    # if the matched probability is
                    if percent > max_percent:
                        # change the original value with the corrected matched value
                        # string_words[i] = name
                        tempName = name

                    # change the max percent to the current matched percent
                    max_percent = percent

            if tempName != '':
                string_words[i] = tempName

        # return the corrected string
        return " ".join(string_words)


# # remove stopwords from line
# def removeStopwords(line):
#     STOPWORDS = ["absolute", "count", "total"]
#     line = re.sub(r'\s*', ' ', line)
#     split_token = ' '  # change based on split token ex:$, \n, _
#     tokens = line.split(split_token)
#     out_line = [w for w in tokens if w not in STOPWORDS]  # delete stopwords from text
#     out_line = ' '.join(out_line)
#     return out_line


def removeStopwordsFromLine(text):
    # ------------------modified----------------------
    # note : momken nsheel kl el : eza l2enaha mabthm
    # note : yle 3 ranges ma7teton lsa lnshof eza bdna yahon
    STOPWORDS = ["absolute", "count", "total", "(total)","(fasting)", "(calculated)", 'h', 'l', "optimum:", "premenopausal:"]
    # ------------------modified----------------------
    tokens = text.split()
    out_text =[w for w in tokens if not w in STOPWORDS]# delete stopwords from text
    out_text = ' '.join(out_text)
    text = out_text
    return out_text

# def removeStopwordsFromText(text):
#     out_text = ""
#     lines = text.splitlines()
#     for line in lines:
#         out_text += removeStopwordsFromTextAssistant(line)
#         out_text += "\n"
#     return out_text

# new nikos to clean text while looping over the lines once
## removed removeStopWordsFormText and removeSideTitleAndComments
## using removeStopWordsFromLine AKA removeStopWordsFromTextAssistant previosly .....  do we need removeStopwords too ????
## we can remove this if we are reading from the files always
def cleanLines(text):
    out_text = ""
    lines = text.splitlines()
    for line in lines:
        out_text += cleanLineAssistant(line)
    return out_text

def cleanLineAssistant(line):
    if not isSidetitleOrComment(line):
        return removeStopwordsFromLine(line) + '\n'
    return ''

# reading file and removing stopwords from it
## comments in function to edit it later
############ never mind this is not even used at all
# def readFile(fileName):
#     lines = []
#     with open(fileName, 'r', encoding='utf-8') as f:
#         for line in f:
#             lines.append(removeStopwords(line.lower()))
#             ### if we are readingg from file we can remove cleanLines and insted use the following to loop less over the lines
#             # lines.append(cleanLineAssistant(line.lower()))
#
#     return lines

# ---------------------------------new and modified (start)----------------------------------
# this part is for extracting columns titles ROW and correcting them then returns a list with the titles found
# our main titles: ["test","result","previousresult","unit","range","lastdatedresult","date"]

# define lists we need
# returns list of lists
def defineOurLists():
    testWords = ["test", "tests", "testname", "parameter", "chemistry"]
    resultWords = ["result", "results"]
    prevResultWords = ["previousresult", " prv.rslt"]
    unitWords = ["unit", "units", "unite"]
    rangWords = ["range", "ranges", "ref.range", "refrange ", "referencevalues", "referencerange",
                 "normalrange", " normalranges", "normalvalue", "normal", "biologicalreferenceintervals",
                 "limit", "expectedvalues"]
    # a7yanan swa a7yanan l7al 1 column or 2 7sab el template.
    lastRes = ["lastdatedresult", "lasttest", "latestresult", "latestresults"]
    date = ["date"]
    wordsList = [testWords, resultWords, prevResultWords, unitWords, rangWords, lastRes, date]

    return wordsList

## replaced with fixing min extract
# remove new lines after and before titles used in miniExtract()
# this function works for before and after new lines we just reverse text and word to do so
# parameters: text: a string the we want to extract headers from, word: a word from our titles list
# returns text after removing new lines (before or after) and a boolean to know if it was changed
# def removeNewLBandA(text,word):
#     # to know if the text changed or not
#     boool = False
#     # get the index where we found the title
#     index = text.find(rf'{word}')
#     # if a title was found search for the next new line
#     if index != -1:
#         indexNewL = text[index:].find('\n')
#         # if new line was found change text from the start to the new line index
#         if indexNewL != -1:
#             indexNewL += len(text[:index])
#             text = text[:indexNewL]
#             boool = True
#             return text, boool
#     # return same text if nothing was found
#     return text, boool

# remove new lines after and before titles used in extractHeadersRow()
# we use this function to call removeNewLBandA() for before and after new lines
# parameters: text: a string the we want to extract headers from, wordsList: our titles lists
#             rev: boolean to use it to remove new lines one time before titles and one after titles
# returns text after removing new lines
# def miniExtract(text,wordsList,rev):
#     # if we want to remove new lines before we reverse text and word then reverse the final text to return to normal
#     if rev:
#         text = text[::-1]
#
#     for subWords in wordsList:
#         for word in subWords:
#             if rev:
#                 word = word[::-1]
#             text, boool = removeNewLBandA(text, word)
#             if boool:
#                 if rev:
#                     return text[::-1]
#                 return text
#     if rev:
#         return text[::-1]
#     return text

# nikos edit
## used to be called mini extract
# parameters: text: a string the we want to extract headers from, wordsList: our titles lists
# returns columns titles ROW
def extractHeadersRow(text, wordList):
    for subwords in wordList:
        for word in subwords:
            index = text.find(rf'{word}')
            if index != -1:
                indexFirstNewL = text[:index].rfind('\n')
                indexLastNewL = text[index:].find('\n')+index


                # print(text[indexFirstNewL+1:indexLastNewL])
                return text[indexFirstNewL+1:indexLastNewL], indexFirstNewL+1


# use miniExtract() to get the final titles row
# parameters: text: a string the we want to extract headers from, wordsList: our titles lists
# returns columns titles ROW
# def extractHeadersRow(text,wordsList):
#     # convert text to lower case
#     text = text.lower()
#     # remove new lines AFTER columns titles
#     text, indexFirstNewL = miniExtract(text,wordsList)
#     # remove new lines BEFORE columns titles
#     # text = miniExtract(text,wordsList,True)
#
#     return text, indexFirstNewL

# used in columnsHeaders() to correct titles spelling and return one name for each found in a list
# our main titles: ["test","result","previousresult","unit","range","lastdatedresult","date"]
# parameters: line: columns titles ROW, wordsList: our titles lists
# returns a list of the titles we found after correction

def getColumnsHeaders(line, wordsList, oldist=None):
    
    headers = []
    length = 5
    ##### may need to fix it to take the header even if it's not in the headers list
    for header in line :
        head = ''
        mini = len(header)/2
        for subWords in wordsList:
            for word in subWords:
                dist = oldist(header, word)
                if dist < length:
                    if dist < mini:
                        mini = dist
                        # to always replace with the same word
                        head = subWords[0]
                else:
                    continue

        if head is not '':
            headers.append(head)

    return headers

# call functions to get the final list of the titles we found
# parameters: contents: ocr output (string), wordsList: our titles lists, splitString: our split marker
# returns the final list of the titles we found
def columnsHeaders(contents,wordsList, splitString):
    # extracting columns titles row
    contents, indexFirstNewL = extractHeadersRow(contents,wordsList)
    # split between " $ " for divided templates if we used spaces it needs modifications
    contentslist = contents.split(splitString)
    # contentslist = contents.split(" $ ")
    # remove spaces from the list in case ocr put lots of spaces between words
    contentslist = [re.sub(r'\s', '', word) for word in contentslist]
    # get headers
    headers = getColumnsHeaders(contentslist,wordsList)

    return headers, indexFirstNewL

# main for extracting columns titles ROW
# we use this function to use every thing in this part
# defines our list then gets the list of titles
# parameters: text: OCR output (string), splitString: our split marker
# returns list of titles found
def Headers(text, splitString):
    wordsList = defineOurLists()
    headersList, indexFirstNewL = columnsHeaders(text,wordsList, splitString)

    # print(indexFirstNewL)

    return headersList, text[indexFirstNewL:]
# ---------------------------------new and modified (end)----------------------------------

# remove side titles and comments
# def removeSideTitelsAndComments(lines):
#     noSideTitlesList = []
#     for line in lines:
#         # l = re.sub(r"[0-9]", '', line)
#         # if line == l:
#         #     #print("this is a side title")
#         #     continue
#         if re.search(r"[0-9]", line):
#             noSideTitlesList.append(line)
#
#     return noSideTitlesList


# check if it is a sideTitle or comment
def isSidetitleOrComment(line):
    return not re.search(r'[0-9]', line)


def splitLists(text, headers):
    linesList = []
    for line in text.splitlines():

        tempLine = {headers[idx]: subWord for idx, subWord in enumerate(line.split(" $ "))}
        linesList.append(tempLine)

    return linesList







########################
###
###
# form here is the code for the first cycle only that will be replaced
# might use it later on for the unknown templates
###
###
########################



# read the biomarker and write it in camel case, arrange range
def getBioMarker(noSideTitlesList):
    li = []
    bye = False
    for l in noSideTitlesList:
        print(l)
        single = l.split()
        #         if single[0]:
        #             bye=False
        # if bye:
        #     continue
        #
        l = capwords(l)
        # if "Comment" in l:
        #     bye = True
        #     print('havana')
        #     continue
        # print(l)
        l = re.sub(r"([a-z])\s([A-Z])", r'\1\2', l)
        #print(l)
        l = re.sub(r'([0-9])\s*-\s*([0-9])', r'\1-\2', l)
        li.append(l)
    return li


def biomarkerDists(biomarkerName):
    mini = 15
    minIDs = []
    minStuff = []
    tempResult = executeString("select id,variablename,symbol from biomarker")
    length = len(biomarkerName)/2
    if length < 0:
        length = 0
    for r in tempResult:
        for r2 in r:
            dist = min(oldist(str(r2[2]).lower(), biomarkerName.lower()), oldist(str(r2[1]).lower(), biomarkerName.lower()))
            if dist < length:
                if dist < mini:
                    mini = dist
                    minIDs.clear()
                    minStuff.clear()
                if dist == mini:
                    minIDs.append(r2[0])
                    minStuff.append(r2)
    return minIDs, mini, minStuff


def createCouples(lines,headersList):
    unitIndex = -1
    tt = []
    #print(lines)
    i=0
    for head in headersList:
        if head != '':
            tt.append(head)
            if 'unit' in head:
                #print(i)
                unitIndex = i
            i+=1

    length = len(lines)
    # print(length)
    couples = []

    for line in range(0, length):
        #print(lines[line])
        temp = []
        i = 0

        ###########
        ## to find titles
        ###########
        # if lines[line] == 'Hemoglobin':
        #     print("title was found")
        #     continue
        words = lines[line].split()
        w = re.sub(r'([a-z])([A-Z])', r'\1 \2', words[0])
        # biomarkersID = executeString('select id from biomarker where lower(symbol)=\'' + (w.split())[0].lower() +'\' or lower(variablename)=\'' + (w.split())[0].lower() + '\'')
        mini = 1000
        biomarkersID = []
        for subWord in w.split():
            tempBiomarkersID, tempMin, stuff = biomarkerDists(subWord.lower())
            if tempMin < mini:
                mini = tempMin
                biomarkersID = tempBiomarkersID

        biomarkersID = set(biomarkersID)
        # if len(biomarkersID) == 0:
        #     continue

        for word in words:
            if i > 3:
                break
            if i < 3:
                temp.append([word, tt[i]])

            if i == 3:
                word = re.sub(r'-', ' ', word)
                word = word.split()
                # print(word[0])
                #print(word)
                num1 = float(word[0])
                num2 = float(word[1])
                if num1 > num2:
                    temp.append([word[0], 'max'])
                    temp.append([word[1], 'min'])
                else:
                    temp.append([word[1], 'max'])
                    temp.append([word[0], 'min'])

            i += 1
        tempID = []
        for id in biomarkersID:
            result = executeString('select symbol from biomarkerUnit where biomarker=\'' + str(id) + '\'')
            for r in result:
                for r2 in r:

                    if str(r2[0]).lower() == temp[unitIndex][0].lower():
                        #print(str(r2[0]).lower(), temp[unitIndex][0].lower(), id)
                        tempID.append(id)
        if len(tempID) != 0:
            tempID = set(tempID)
            temp.append([tempID, 'ID'])
        else:
            temp.append([biomarkersID, 'No ID unit'])

        if(len(stuff) > 0):
            temp.append([stuff[0], 'correction'])
        couples.append(temp)

    return couples
















# importing package to use sqlite database
import sqlite3
# importing package to read from directory
import glob
import os

# to store the cursor of the connection to execute the sql command
cursors = []
# to store connections with multiple database files
connections = []

#connect to DB
def connectDB(fileName):

    global cursors
    global connections
    cursors = []
    connections = []

    # initializing the connections and the cursors for the database
    for filename in glob.glob(os.path.join("NLP/MetadataDatabases_1.0", fileName)):
        conn = sqlite3.connect(filename)
        connections.append(conn)
        cur = conn.cursor()
        cursors.append(cur)
        conn.text_factory = lambda b: b.decode(errors='ignore')


# function to execute sql command and return the result
def executeString(s):
    result = []
    for cur in cursors:
        cur.execute(s)
        rows = cur.fetchall()
        result.append(rows)
    return result

def disconnect():
    for conn in connections:
        conn.close()

def createCheckerFile(keyName):
    words = ''

    res = executeString('select variableName from ' + keyName)

    for r in res:
        for row in r:
            line = row[0].split(' ')
            for word in line:
                words = words + word + ','

    text_file = open('NLP/' + keyName + '.txt', 'w', encoding='utf8')
    text_file.write(words)
    text_file.close()


def process_image(image_path):
    """
    High-level function to process an image.
    Orchestrates the flow of image processing, OCR, and text extraction.
    """

    # Step 1: Load and process image (remove white spaces, rotation, etc.)
    print("Processing image...")
    image = load_and_process_image(image_path)

    # Step 2: Perform OCR on processed image
    print("Performing OCR...")
    ocr_text = perform_ocr(image)

    # Step 3: Process the OCR text (cleaning, spell-checking, etc.)
    print("Cleaning and processing OCR text...")
    headers, cleaned_text = process_text(ocr_text)

    # Step 4: Spell check and finalize output
    print("Spell checking and normalizing...")
    lines_list = perform_spell_check_and_normalization(cleaned_text, headers)

    # Return the processed results
    return lines_list, headers


def load_and_process_image(image_path):
    """
    Load the image and apply pre-processing steps like white detection.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at path {image_path}")

    # Apply white detection (remove noise or unwanted white space)
    img = whiteDetection(img)

    # Save the processed image (if necessary, can be omitted)
    processed_image_path = 'OCR/processed_image.jpg'
    cv2.imwrite(processed_image_path, img)

    return img


def perform_ocr(image):
    """
    Perform OCR on the image to extract text.
    """
    Split.main('OCR/processed_image.jpg')  # Split the image if needed
    ocr_output = OCRforCroppedImages('OCR/images')

    return ocr_output


def process_text(ocr_text):
    """
    Process the extracted OCR text (lowercase, headers separation, etc.)
    """
    ocr_text = ocr_text.lower()

    # Split the text into headers and main content
    result = NLP.Headers(ocr_text, " $ ")

    # Print the result to debug and understand the output
    print("Headers result:", result)

    # Unpack into two values (headers and text)
    if isinstance(result, tuple) and len(result) == 2:
        headers, text = result
    else:
        print(f"Unexpected return value from NLP.Headers(): {result}")
        return [], []  # Return empty lists if unexpected value

    # Clean the text further if necessary
    cleaned_text = NLP.cleanLines(text)

    return headers, cleaned_text


def perform_spell_check_and_normalization(text, headers):
    """
    Perform spell check and normalization on the extracted text.
    """
    # Example: Split the cleaned text into lines
    lines_list = NLP.splitLists(text, headers)

    # Perform spell-checking or other normalizations
    Spell_Checker.rangeNormlizer(lines_list)

    return lines_list


def format_result_as_json(lines_list, headers):
    """
    Format the results as a structured JSON-like output.
    """
    final_string = '{\n\n\n'
    for line in lines_list:
        for head in headers:
            if head != 'lastdatedresult':
                final_string += f'\t{head} : {line[head]}\n'
        final_string += '\n\n'
    final_string += '}'
    return final_string


if __name__ == "__main__":
    image_path = 'Katrangi.jpg'  # Replace with your actual image path

    try:
        lines_list, headers = process_image(image_path)
        formatted_result = format_result_as_json(lines_list, headers)
        print(formatted_result)
    except Exception as e:
        print(f"An error occurred: {e}")
