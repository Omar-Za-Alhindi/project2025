from NLP import Spell_Checker, NLP
from NLP import DP_connector
import cv2
from OCR.whiteDetection import whiteDetection
from OCR.OCRforCroppedImages import OCRforCroppedImages
from OCR import Split
import glob
import os


result = ''

def main(imageName):

    # image = cv2.imread(imageName)
    #
    # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    #
    # cv2.imwrite(imageName, image)


    for filename in glob.glob(os.path.join("OCR/images", '*')):
        os.remove(filename)
    img = cv2.imread(imageName)
    img = whiteDetection(img)
    cv2.imwrite('OCR/hello.jpg', img)
    Split.main('OCR/hello.jpg')
    OCRoutput = OCRforCroppedImages('OCR/images')
    with open('test.txt', 'w', encoding='utf8') as f:
        f.write(OCRoutput)

    # DP_connector.connectDB('*')

    with open('test.txt', 'r', encoding='utf8') as f:
        contents = f.read()

    contents = contents.lower()

    headers, text = NLP.Headers(contents, " $ ")
    # contents = contents.splitlines()
    # for line in contents:
    #     print(NLP.removeStopwordsFromTextAssistant(line))


    text = NLP.cleanLines(text)

    # text = text.splitlines()

    # allLines = NLP.removeSideTitelsAndComments(text)
    # allLines = NLP.getBioMarker(allLines)

    # couples = NLP.createCouples(allLines, headers)
    print(headers)
    print('\n\n')
    print(text)
    # print(contents)

    linesList = NLP.splitLists(text, headers)

    Spell_Checker.dropLastDatedResult(linesList)
    # Spell_Checker.unitCorrection(linesList)
    Spell_Checker.rangeNormlizer(linesList)
    # Spell_Checker.biomarkerFixer(linesList)

    for line in linesList:
        print(line)

    global result
    result = resultString(linesList, headers)

    # DP_connector.disconnect()

    return linesList, headers

def returnResult():
    return result


def resultString(result, headers):

    finalString = '{\n\n\n'
    for line in result:
        for head in headers:
            if head != 'lastdatedresult':
                finalString += '\t' + head + ' : ' + line[head] + '\n'
        finalString += '\n\n'
    finalString += '}'
    return finalString


if __name__ == "__main__":
    image_name = "Katrangi.jpg"  # Replace this with the actual path to the image you want to process
    linesList, headers = main(image_name)

    # You can now use linesList and headers for further processing or output
    print("Processed Lines:")
    for line in linesList:
        print(line)

    print("\nHeaders:")
    print(headers)
