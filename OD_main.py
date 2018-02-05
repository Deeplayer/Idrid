import glob, cv2, time
from OD_segmentation import *
import numpy as np
import matplotlib.pyplot as plt



EX_AR_PATH = "../Idrid/EX_seg_set/Training Set/Apparent Retinopathy/**.jpg"
OD_MASK_PATH = "../Idrid/OD Segmentation Training Set/**.tif"
AR_PATH = glob.glob(EX_AR_PATH)
OD_PATH = glob.glob(OD_MASK_PATH)
D = 0
J = 0
for j in range(0,len(AR_PATH)):
    size = 720
    img = cv2.imread(AR_PATH[j])
    print(AR_PATH[j])
    #img = cv2.imread("../Idrid/10_right.jpeg")
    GT = cv2.cvtColor(cv2.imread(OD_PATH[j]), cv2.COLOR_BGR2GRAY)
    GT[GT>0] = 1.
    GT[GT<=0] = 0.
    plt.subplot(121)
    plt.imshow(GT)
    t0 = time.time()
    mask, _, _ = OD_seg(img)
    #mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # plt.imshow(mask)
    # plt.show()
    t1 = time.time()

    mask[mask>0] = 1.
    mask[mask<=0] = 0.
    assert (GT.shape==mask.shape)
    plt.subplot(122)
    plt.imshow(mask)
    Jaccard = np.sum(GT*mask)/(np.sum(GT)+np.sum(mask)-np.sum(GT*mask))
    Dice = 2*np.sum(GT*mask)/(np.sum(GT)+np.sum(mask))
    D += Dice
    J += Jaccard
    print(Dice, Jaccard, (t1-t0))
    #plt.show()
    plt.close()

print()
print(D/54, J/54)      # D = 0.944   J = 0.896  Specificity[TP/(TP+FP)] = 1  Sensitivity[TP/(TP+FN)] = 1
#                        # time: ~2 s

