import matplotlib.pyplot as plt
import numpy as np
import cv2, math


def blobNum(img):
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img.astype(np.uint8))
    nb_components = nb_components - 1

    return nb_components


def findMaxblob(img):
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img.astype(np.uint8))
    sizes = stats[1:, -1]
    if len(sizes) <= 1:
        return img
    idx = np.argsort(sizes)[::-1]
    mask = np.zeros((output.shape)).astype(np.uint8)
    mask[output == idx[0]+1] = 255

    return mask


def findMaxblob2(img):
    nb_components, output, stats, loc = cv2.connectedComponentsWithStats(img.astype(np.uint8))
    w = img.shape[0]
    loc = loc[1:]
    sizes = stats[1:, -1]
    idx = np.argsort(sizes)[::-1][0:2]
    d1 = min(loc[idx[0]][0], loc[idx[0]][1], w-loc[idx[0]][0], w-loc[idx[0]][1])
    #d2 = min(loc[idx[1]][0], loc[idx[1]][1], w - loc[idx[1]][0], w - loc[idx[1]][1])
    blob1 = np.zeros((output.shape)).astype(np.uint8)
    blob1[output == idx[0] + 1] = 255
    blob2 = np.zeros((output.shape)).astype(np.uint8)
    blob2[output == idx[1] + 1] = 255
    if d1 > 180 and d1 < 330 and sizes[idx[1]] > 2500 and circularity(blob2) < 1.23:
        return blob2
    else:
        return blob1


def coordinateMask(img):
    _, _, stats, _ = cv2.connectedComponentsWithStats(img.astype(np.uint8))

    # location of the object
    yu, yd, xl, xr = stats[1][1], stats[1][1]+stats[1][3], stats[1][0], stats[1][0]+stats[1][2]
    if yd-yu > xr-xl:
        xl -= ((yd-yu)-(xr-xl))//2
        xl = max(0, xl)
        xr += ((yd-yu)-(xr-xl))//2
        xr = min(img.shape[1], xr)

    binary = img[yu:yd, xl:xr]

    return binary, [yu, yd, xl, xr]


def locOD(img):
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img.astype(np.uint8))
    loc = _[1:][0]
    return loc


def clahe_equalized(img, cl=1.0):
    # create a CLAHE object (Arguments are optional)
    clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(8,8))
    img_equalized = clahe.apply(img)

    return img_equalized


def kernel(size=7):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def kernel2(size=6):
    k = np.array([[0,0,1,1,0,0],
                  [0,1,1,1,1,0],
                  [0,1,1,1,1,0],
                  [0,1,1,1,1,0],
                  [0,1,1,1,1,0],
                  [0,0,1,1,0,0]]).astype(np.uint8)

    return k


def threshold(img, v):
    _, THr = cv2.threshold(img.astype(np.uint8), np.mean(img) + v*np.std(img), 255, cv2.THRESH_BINARY)
    return THr


def circularity(img):
    _, contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 5
    else:
        maxc = contours[0]
    area = cv2.contourArea(maxc)
    perimeter = cv2.arcLength(maxc, True)
    circularity = perimeter ** 2 / ((4 * math.pi * area)+1e-4)

    return circularity


def contour(img):
    _, contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxc = contours[0]
    return maxc



def OD_seg(img):

    size = 520
    kappa = 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ker = min(gray.shape[0], gray.shape[1]) // 50
    if ker % 2 == 0: ker += 1

    gray_1 = cv2.medianBlur(gray, ker)

    _, TH = cv2.threshold(gray_1, max(0, np.mean(gray_1) - kappa*np.std(gray_1)), 255, cv2.THRESH_BINARY)
    while circularity(TH) > 1.16 or circularity(TH) < 1:
        th = max(np.mean(gray_1) - kappa * np.std(gray_1), 0)
        if th <= 0:
            kappa -= 0.1
        else:
            kappa += 0.1
        _, TH = cv2.threshold(gray_1, th, 255, cv2.THRESH_BINARY)

    binary, loc = coordinateMask(TH)
    img_crop = img[loc[0]:loc[1], loc[2]:loc[3]]

    W, H = img_crop.shape[1], img_crop.shape[0]
    img_1 = cv2.resize(img_crop, (size, size))
    binary_1 = cv2.resize(binary, (size, size))

    img_1[:, :, 0][binary_1 == 0] = np.mean(img_1[:, :, 0])   # B channel
    img_1[:, :, 1][binary_1 == 0] = np.mean(img_1[:, :, 1])   # G channel
    img_1[:, :, 2][binary_1 == 0] = np.mean(img_1[:, :, 2])   # R channel

    img_ref = cv2.addWeighted(img_1, 4, cv2.GaussianBlur(img_1, (0, 0), 30), -4, 100)

    img_r = img_1[:,:,2]    # R channel

    if np.mean(img_r) > 230:
        img_r = img_ref[:,:,1]
        img_r = clahe_equalized(img_r, cl=2.5)
        img_r[binary_1 == 0] = 0
    else:
        img_r = clahe_equalized(img_r, cl=2.5)
        img_r[binary_1 == 0] = 0
        img_r = cv2.medianBlur(img_r, 31)  # 31

    # plt.imshow(img_r, cmap='gray')
    # plt.show()

    # estimated OD area
    area_OD = math.pi*45**2
    v = 1.4
    THr = threshold(img_r, v)

    while np.sum(THr)/255 < 0.8*area_OD:
        v -= 0.05
        THr = threshold(img_r, v)

    poch = 5
    tp = THr
    while circularity(findMaxblob(tp)) > 1.4:
        if poch <= 0:
            break
        else:
            v += 0.05
            tp = threshold(img_r, v)
            poch -= 1

    if np.sum(tp)/255 < 0.6*area_OD:
        thr = THr
    else:
        thr = tp

    # plt.imshow(thr, cmap='gray')
    # plt.show()

    p0 = 15
    while circularity(findMaxblob(thr)) > 1.3:
        thr = cv2.morphologyEx(THr, cv2.MORPH_OPEN, kernel(p0))
        p0 += 2

    # plt.imshow(thr, cmap='gray')
    # plt.show()

    if blobNum(thr) > 1:
        thr = findMaxblob2(thr)

    # plt.imshow(thr, cmap='gray')
    # plt.show()

    loc_od = locOD(thr)
    img_ref[:, :, 2][binary_1==0] = 0
    xl, xr = max(0,int(loc_od[0])-80), min(size,int(loc_od[0])+80)
    yu, yd = max(0,int(loc_od[1])-80), min(size,int(loc_od[1])+80)
    bench1 = cv2.medianBlur(img_ref[:,:,2][yu:yd,xl:xr], 9)
    bench2 = img_r[yu:yd, xl:xr]

    # plt.figure(figsize=(13, 8))
    # plt.subplot(231)
    # plt.imshow(bench1, cmap='gray')
    # plt.subplot(232)
    # plt.imshow(bench2, cmap='gray')

    beta1 = 0.9
    beta2 = 0.6
    th1 = np.mean(bench1)+beta1*np.std(bench1)
    th2 = np.mean(bench2)+beta2*np.std(bench2)
    while th1 >= 250 or th2 >= 250:
        beta1 -= 0.1
        beta2 -= 0.1
        th1 = np.mean(bench1) + beta1 * np.std(bench1)
        th2 = np.mean(bench2) + beta2 * np.std(bench2)


    seg1 = threshold(bench1, beta1)
    seg2 = threshold(bench2, beta2)

    iters = 20
    while circularity(findMaxblob(seg1)) > 1.3 and iters > 0:
        beta1 += 0.05
        seg1 = threshold(bench1, beta1)
        iters -= 1

    if iters == 0 and circularity(findMaxblob(seg1)) > 1.3:
        seg1 = threshold(bench1, beta1-0.8)

    # process seg1:
    seg1 = findMaxblob(seg1)
    seg1 = cv2.morphologyEx(seg1, cv2.MORPH_OPEN, kernel(23))     # 23
    seg1 = cv2.morphologyEx(seg1, cv2.MORPH_CLOSE, kernel(25))    # 25
    seg1 = cv2.erode(seg1, kernel(3), iterations=1)
    seg1[binary_1[yu:yd,xl:xr]==0] = 0
    #seg1 = cv2.dilate(seg1, kernel(5), iterations=1)
    # plt.imshow(seg1)
    # plt.show()

    tp = seg1
    if circularity(seg1) > 1.2:
        p1, poch1 = 31, 30
        while circularity(tp) > 1.2 and p1 > 1 and poch1 > 0 :
            tp = cv2.morphologyEx(seg1, cv2.MORPH_OPEN, kernel(p1))
            poch1 -= 1
            if np.sum(tp)/255 < 0.7*area_OD:
                p1 -= 2
            else:
                p1 += 2

    seg1 = tp

    if np.sum(seg1)/255 == 0:
        cir1 = 5
    else:
        cir1 = circularity(seg1)

    # process seg2:
    seg2 = cv2.morphologyEx(seg2, cv2.MORPH_OPEN, kernel(15))
    seg2 = findMaxblob(seg2)

    temp = seg2
    p2, poch2 = 31, 20
    while circularity(findMaxblob(temp)) > 1.18 and poch2 > 0 and p2 > 1:
        temp = cv2.morphologyEx(seg2, cv2.MORPH_OPEN, kernel(p2))
        poch2 -= 1
        if np.sum(findMaxblob(temp))/255 < 0.5*area_OD:
            p2 -= 2
        else:
            p2 += 2

    seg2 = findMaxblob(temp)
    if (np.sum(seg2)/255) > 1.5*area_OD:
        seg2 = cv2.morphologyEx(seg2, cv2.MORPH_OPEN, kernel(75))
        seg2 = cv2.erode(seg2, kernel(5), iterations=2)

    # process seg3:
    if cir1 < 1.3 and np.sum(seg1)/255 < 1.8*area_OD and blobNum(seg1) == 1 and np.sum(seg1)/255 > 0.5*area_OD:
        seg = seg1

    else:
        p3, poch3 = 31, 40
        temp = thr
        while circularity(temp) > 1.15 and poch3 > 0 and p3 > 1:
            temp = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel(p3))
            poch3 -= 1
            if np.sum(temp)/255 < 0.5*area_OD:
                p3 -= 2
            else:
                p3 += 2

        if np.sum(temp)/255 > 1.5*area_OD:
            temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel(77))
            temp = cv2.erode(temp, kernel(5), iterations=3)
            temp = cv2.erode(temp, kernel(3), iterations=1)

        if np.sum(temp)/255 < 0.7*area_OD:
            temp = cv2.dilate(temp, kernel2(6), iterations=4)

        seg3 = temp[yu:yd, xl:xr]

        if circularity(seg2) > 1.2 or np.sum(seg2)/255 >= 1.7*area_OD:
            seg = seg3
        elif np.sum(seg3) >= 0.8*np.sum(seg2) and circularity(seg3) <= 1.18:
            seg = seg3
        else:
            seg = seg2

    if np.sum(seg)/255 > 1.5*area_OD:
        seg = cv2.erode(seg, kernel(3), iterations=1)

    # plt.subplot(233)
    # plt.imshow(seg1, cmap='gray')
    # plt.subplot(234)
    # plt.imshow(seg2, cmap='gray')
    #
    # plt.subplot(235)
    # plt.imshow(seg, cmap='gray')
    # plt.subplot(236)

    seg_res = np.zeros((size,size)).astype(np.uint8)
    seg_res[yu:yd,xl:xr] = seg

    img_1[binary_1 == 0] = 0
    cv2.drawContours(img_1, contour(seg_res), -1, (0, 255, 0), 3)
    # plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
    # #plt.show()
    # plt.close()
    seg_res[binary_1==0] = 0
    seg_res = cv2.resize(seg_res, (W, H))

    mask = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
    mask[loc[0]:loc[1],loc[2]:loc[3]] = seg_res

    return mask, binary, loc
    #
    # #cv2.imwrite(name1 + '.jpg', img_ot)

