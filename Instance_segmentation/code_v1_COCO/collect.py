import os

import numpy
import cv2
import numpy as np

dir1 = "/home/park/PycharmProjects/PolarMask-master/work_dirs_polar_768_1x_r50_org_lr_down/test_vis/comparison/"
dir2 = "/home/park/PycharmProjects/Eigencontours/work_dirs_polar_768_1x_r50_360_36_lr_down/test_vis/comparison/"

set1 = set(os.listdir(dir1))
set2 = set(os.listdir(dir2))
com_list = list(set1.intersection(set2))

def mkdir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)

mkdir("comparison")

for i in com_list:
    img1 = cv2.imread(dir1 + i)
    img2 = cv2.imread(dir2 + i)
    new = np.concatenate((img1[:, :1280, :], img2[:, :2560, :]), axis=1)
    cv2.imwrite("comparison/" + i, new)