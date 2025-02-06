import cv2
import glob
import timeit

shami = cv2.imread("OCR/Template Matching/Shami.jpg", 0)
katrangi = cv2.imread("OCR/Template Matching/Katrangi.jpg", 0)
wasara = cv2.imread("OCR/Template Matching/wasara.jpg" ,0)
teshren = cv2.imread("OCR/Template Matching/Teshren.jpg" , 0)
abnsena = cv2.imread("OCR/Template Matching/Abnsena.jpg" ,0)
katrangi = cv2.resize(katrangi, (400, 400), interpolation=cv2.INTER_AREA)
shami = cv2.resize(shami, (400, 400), interpolation=cv2.INTER_AREA)
wasara = cv2.resize(wasara, (400, 400), interpolation=cv2.INTER_AREA)
teshren = cv2.resize(teshren, (400, 400), interpolation=cv2.INTER_AREA)
abnsena = cv2.resize(abnsena, (400, 400), interpolation=cv2.INTER_AREA)

finder = cv2.xfeatures2d.SIFT_create()
kp_shami, des_shami = finder.detectAndCompute(shami, None)
kp_katrangi, des_katrangi = finder.detectAndCompute(katrangi, None)
kp_wasara, des_wasara = finder.detectAndCompute(wasara, None)
kp_teshren, des_teshren = finder.detectAndCompute(teshren, None)
kp_abnsena, des_abnsena = finder.detectAndCompute(abnsena, None)

lowe_ratio = 0.5
bf = cv2.BFMatcher()
def tm(image):
    img = cv2.imread(image)
    img1 = cv2.imread(image, 0)
    img1 = cv2.resize(img1, (700, 700), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (700, 700), interpolation=cv2.INTER_AREA)
    kp1, des1 = finder.detectAndCompute(img1, None)
    matches_shami = bf.knnMatch(des1,des_shami, k=2)
    matches_katrangi = bf.knnMatch(des1 , des_katrangi ,k=2)
    matches_wasara = bf.knnMatch(des1, des_wasara, k=2)
    matches_teshren = bf.knnMatch(des1, des_teshren, k=2)
    matches_abnsena = bf.knnMatch(des1 ,des_abnsena ,k=2)

    s = []
    k = []
    w = []
    t = []
    a = []
    final = []
    for m1,n1 in matches_shami:
        if m1.distance < lowe_ratio*n1.distance:
            s.append([m1])
    final.append((len(s) ,'S'))
    for m2,n2 in matches_katrangi:
        if m2.distance < lowe_ratio*n2.distance:
            k.append([m2])
    final.append((len(k) ,'K'))
    for m3,n3 in matches_wasara:
        if m3.distance < lowe_ratio*n3.distance:
            w.append([m3])
    final.append((len(w) , 'W'))
    for m4,n4 in matches_teshren:
        if m4.distance < lowe_ratio*n4.distance:
            t.append([m4])
    final.append((len(t) , 'T'))
    for m5,n5 in matches_abnsena:
        if m5.distance < lowe_ratio*n5.distance:
            a.append([m5])
    final.append((len(a) , 'A'))
    final.sort()
    # cv2.imshow('ada', img)
    # cv2.waitKey(100)
    # print(final[-1][0])
    if final[-1][1] == 'K' and final[-1][0]>=15:
        # print('ktrangiii')
        return final[-1][1]

    elif final[-1][1] == 'S' and final[-1][0]>=20:
        return final[-1][1]
    elif final[-1][1] == 'W' and final[-1][0]>=15:
        return final[-1][1]
    elif final[-1][1] == 'T' and final[-1][0]>=15:
        return final[-1][1]
    elif final[-1][1] == 'A' and final[-1][0]>=15:
        return final[-1][1]
    else:return 'None'





# template = []
# template.append((des_katrangi ,'K'))
# template.append((des_shami ,'S'))
# # print(template)
# print(len(template))
# lowe_ratio = 0.5
# def tm(image):
#     img = cv2.imread(image)
#     img1 = cv2.imread(image, 0)
#     img1 = cv2.resize(img1, (700, 700), interpolation=cv2.INTER_AREA)
#     img = cv2.resize(img, (700, 700), interpolation=cv2.INTER_AREA)
#     kp1, des1 = finder.detectAndCompute(img1, None)
#     bf = cv2.BFMatcher()
#     good = []
#     final = []
#     max_matches = 0
#     j = 0
#     final_template = ''
#
#     for i in template:
#
#         matches = bf.knnMatch(des1 ,i[0] ,k=2)
#         # if i[1] =='S':
#         #     j = 20
#         # if i[1] =='K':
#         #     j = 15
#         for m, n in matches:
#             if m.distance < lowe_ratio * n.distance:
#                 good.append([m])
#         final.append((len(good), i[1]))
#
#     #             if len(good)> max_matches and len(good) >=j:
#     #                 max_matches = len(good)
#     #                 final_template = i[1]
#     #             else: final_template = 'none'
#     # print(final_template)
#     print(final)
#     cv2.imshow('sd',img)
#     cv2.waitKey(0)
# def tm(image):
#     img = cv2.imread(image)
#     img1 = cv2.imread(image, 0)
#     img1 = cv2.resize(img1, (700, 700), interpolation=cv2.INTER_AREA)
#     img = cv2.resize(img, (700, 700), interpolation=cv2.INTER_AREA)
#     shami = cv2.imread("D:\\5\\Project\\shami\\shamiTest10.jpg", 0)
#     katrangi = cv2.imread("D:\\5\\Project\\Resize\\Katrangi\\NewResults\\New folder\\KatrangiTest11.jpg", 0)
#     katrangi = cv2.resize(katrangi, (400, 400), interpolation=cv2.INTER_AREA)
#     shami = cv2.resize(shami, (400, 400), interpolation=cv2.INTER_AREA)
#     lowe_ratio = 0.5
#     finder = cv2.xfeatures2d.SIFT_create()
#     kp1, des1 = finder.detectAndCompute(img1, None)
#     kp_shami, des_shami = finder.detectAndCompute(shami, None)
#     kp_katrangi, des_katrangi = finder.detectAndCompute(katrangi, None)
#     bf = cv2.BFMatcher()
#     matches_shami = bf.knnMatch(des1,des_shami, k=2)
#     matches_katrangi = bf.knnMatch(des1 , des_katrangi ,k=2)
#     good1 = []
#     good2 = []
#     for m1,n1 in matches_shami:
#         if m1.distance < lowe_ratio*n1.distance:
#             good1.append([m1])
#     for m2,n2 in matches_katrangi:
#         if m2.distance < lowe_ratio*n2.distance:
#             good2.append([m2])
#
#     print('sahmiiii' , len(good1))
#     print('katrnagiii' , len(good2))
    # matches_shami = bf.knnMatch(des1, des_shami, k=2)
    # matches_katrangi = bf.knnMatch(des1, des_katrangi, k=2)
    # template = []
    # good = []
    # final = []
    # template.append((des_katrangi, 'K'))
    # template.append((des_shami, 'S'))
    # for i in template:
    #     matches = bf.knnMatch(des1 ,i[0] ,k=2)
    #     for m, n in matches:
    #         if m.distance < lowe_ratio * n.distance:
    #             good.append([m])
    #     final.append((len(good), i[1]))
    # print(final)

# for image in glob.iglob('D:\\5\Project\\templets\\*.jpg'):
#    x = tm(image)
#    print(x)
# image = "D:\\5\\Project\\HomsTest11.jpg"
# x = tm(image)
# print(x)