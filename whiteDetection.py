import cv2


def whiteDetection(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to the grayscale image
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Correctly unpack two values (OpenCV 4.x)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Optionally, draw the contours on the image (for debugging purposes)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    return img
