''' Crop (remove the whole black space, except corners), 
    Subtract the local average color,
    Resize images to square, save as jpg. '''

import cv2, glob, random, math
import matplotlib.pyplot as plt
import numpy as np


def findMaxblob(img):
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img.astype(np.uint8))
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    mask = np.zeros((output.shape))
    # for every component in the image, keep it only if it's the maximal blob
    for i in range(0, nb_components):
        if sizes[i] == max(sizes):
            mask[output == i + 1] = 255

    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

    # location of the object
    yu, yd, xl, xr = stats[1][1], stats[1][1]+stats[1][3], stats[1][0], stats[1][0]+stats[1][2]
    if yd-yu > xr-xl:
        xl -= ((yd-yu)-(xr-xl))//2
        xl = max(0, xl)
        xr += ((yd-yu)-(xr-xl))//2
        xr = min(mask.shape[1], xr)

    binary = img[yu:yd, xl:xr]

    return binary, [yu, yd, xl, xr]


def findMaxcontour(x):
    maxc = []
    for c in x:
        if len(c) >= len(maxc):
            maxc = c

    return maxc


def scale(img, scale=400):
    x = img[img.shape[0] // 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)


PATH = 'F:/PythonProjects/Idrid/train/**.jpeg'
PATH_OUT = 'F:/PythonProjects/Idrid/processed/'
size_out = 512

for path in glob.glob(PATH):
    print(path)   
    img = cv2.imread(path)
    plt.figure(figsize=(13, 8))
    plt.subplot(231)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('original image')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # segmentation, method-1, has good performance for dark image
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    binary1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, np.ones((40,40),np.uint8))

    # segmentation, method-2, has good performance for bright image
    _1, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    binary2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, np.ones((10,10),np.uint8))

    binary1, loc1 = findMaxblob(binary1)
    binary2, loc2= findMaxblob(binary2)

    _, contours1, _ = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    maxc1 = findMaxcontour(contours1)
    area1 = cv2.contourArea(maxc1)
    perimeter1 = cv2.arcLength(maxc1, True)
    maxc2 = findMaxcontour(contours2)
    area2 = cv2.contourArea(maxc2)
    perimeter2 = cv2.arcLength(maxc2, True)
    circularity1 = perimeter1 ** 2 / (4 * math.pi * area1)
    circularity2 = perimeter2 ** 2 / (4 * math.pi * area2)
    print(circularity1, circularity2)
    print()

    m1 = np.ones((binary1.shape[0],binary1.shape[1],3))
    m2 = np.ones((binary2.shape[0],binary2.shape[1],3))
    cv2.drawContours(m1, maxc1, -1, (0, 255, 0), 20)
    cv2.drawContours(m2, maxc2, -1, (0, 255, 0), 20)
    plt.subplot(232)
    plt.imshow(m1)
    plt.title(str(round(circularity1,3)))
    plt.subplot(233)
    plt.imshow(m2)
    plt.title(str(round(circularity2,3)))
    #plt.show()
    
    ratio1 = area1 / (gray.shape[0]*gray.shape[1])
    ratio2 = area2 / (gray.shape[0]*gray.shape[1])
    if (abs(circularity1-1) < abs(circularity2-1)) and ratio1 < 0.9 or ratio2 < 0.3:
        binary = binary1
        loc = loc1
    else:
        binary = binary2
        loc = loc2

    img = img[loc[0]:loc[1], loc[2]:loc[3]]

    img = scale(img)

    # subtract local mean color
    img_1 = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), img.shape[1]/60), -4, 128)

    # remove 8% outer boundary
    x = np.zeros(img_1.shape)
    cv2.circle(x, (img_1.shape[1] // 2, img_1.shape[0] // 2), int(img.shape[1]*0.46), (1, 1, 1), -1)
    img_2 = img_1 * x + 128 * (1 - x)
    radius = int(img.shape[1]*0.46)
    xl = img_1.shape[1]//2 - radius
    xr = img_1.shape[1]//2 + radius
    if xr-xl > img_1.shape[0]:
        img_2 = img_2[:, xl:xr]
    else:
        yu = img_1.shape[0]//2 - radius
        yd = img_1.shape[0]//2 + radius
        img_2 = img_2[yu:yd, xl:xr]

    img_res = cv2.resize(img_2, (size_out, size_out))

    plt.subplot(234)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(235)
    plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
    plt.subplot(236)
    plt.imshow(cv2.cvtColor(img_res.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.xlabel('processed image (512x512)')
    #plt.show()
    plt.savefig(path[30:-5] + '.png')
    plt.close()

    cv2.imwrite(PATH_OUT + path[30:-5] + '.jpg', img_res)
