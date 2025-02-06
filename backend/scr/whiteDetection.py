import cv2
import numpy as np


# in this scropt we need to have an image to call whiteDetection(img)
# it returns image after detection (without background just the paper)

def whiteDetection(frame):
    # Convert to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White mask
    lower_white = np.array([0, 0, 38])
    upper_white = np.array([180, 77, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Filter
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)

    # Contours detection
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = 0

    # Drawing contour on the paper, then projecting that contour
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        if len(approx) == 4:
            # contour area
            if area > 1000000:
                screenCnt = approx
                frame1 = frame.copy()
                cv2.drawContours(frame1, [screenCnt], -1, (0, 255, 0), 5)

                # reshape to avoid errors ahead
                screenCnt = screenCnt.reshape((4, 2))

                # create a new array and initialize
                new_screenCnt = np.zeros((4, 2), dtype="float32")

                Sum = screenCnt.sum(axis=1)
                new_screenCnt[0] = screenCnt[np.argmin(Sum)]
                new_screenCnt[2] = screenCnt[np.argmax(Sum)]

                Diff = np.diff(screenCnt, axis=1)
                new_screenCnt[1] = screenCnt[np.argmin(Diff)]
                new_screenCnt[3] = screenCnt[np.argmax(Diff)]

                (tl, tr, br, bl) = new_screenCnt

                # find distance between points and get max
                dist1 = np.linalg.norm(br - bl)
                dist2 = np.linalg.norm(tr - tl)

                maxLen = max(int(dist1), int(dist2))

                dist3 = np.linalg.norm(tr - br)
                dist4 = np.linalg.norm(tl - bl)

                maxHeight = max(int(dist3), int(dist4))

                dst = np.array([[0, 0], [maxLen - 1, 0], [maxLen - 1, maxHeight - 1], [0, maxHeight - 1]],
                               dtype="float32")

                N = cv2.getPerspectiveTransform(new_screenCnt, dst)

                # Storing the result
                new_frame = cv2.warpPerspective(frame, N, (maxLen, maxHeight))
                # # save result
                # cv2.imwrite(
                #     "D:\\5th Year\Second Semester\Senior Project\MR Data\\test\\nikos\\Results\\" + "result.jpg",
                #     new_img)

                # # see results
                # # contour
                # frame1 = cv2.pyrDown(frame1)
                # frame1 = cv2.pyrDown(frame1)
                # cv2.imshow("Frame", frame1)
                # # mask
                # mask = cv2.pyrDown(mask)
                # mask = cv2.pyrDown(mask)
                # cv2.imshow("white_mask", mask)
                # # final result
                # new_frame = cv2.pyrDown(new_frame)
                # new_frame = cv2.pyrDown(new_frame)
                # cv2.imshow("result", new_frame)


    return new_frame


if __name__ == '__main__':
    # main
    img = cv2.imread("D:\\5th Year\Second Semester\Senior Project\MR Data\Files\Syria\Damascus\Katrangi\IMG_6926.JPG")
    img = whiteDetection(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
