import os
import cv2
import numpy as np
from tqdm import tqdm

class merging(object):
    def __init__(self):
        super(merging, self).__init__()
        self.h, self.w = (768, 1280)
        self.vis_h, self.vis_w = (384, 640*4)
    def vis(self, dir1, dir2, new_dir=None):
        list1 = os.listdir(dir1)
        list2 = os.listdir(dir2)
        inter = set(list1).intersection(set(list2))
        for i in tqdm(inter):
            img_e = cv2.imread(dir1 + i)
            org, eigen, gt = img_e[:, :self.w], img_e[:, self.w: self.w*2], img_e[:, self.w*2:]
            polar = cv2.imread(dir2 + i)
            new = np.concatenate((org, polar, eigen, gt), axis=1)
            new = cv2.resize(new, (self.vis_w, self.vis_h))
            cv2.imwrite(new_dir + i, new)

    def save(self, dir1, dir2, list, new_dir=None):
        for i in list:
            img_e = cv2.imread(dir1 + i + '.png')
            org, eigen, gt = img_e[:, :self.w], img_e[:, self.w: self.w*2], img_e[:, self.w*2:]
            polar = cv2.imread(dir2 + i + '.png')
            if not os.path.isdir(new_dir + i):
                os.makedirs(new_save + i)
            new_dir_ = new_save + i + '/'
            cv2.imwrite(new_dir_ + 'org.png', org)
            cv2.imwrite(new_dir_ + 'polar.png', polar)
            cv2.imwrite(new_dir_ + 'eigen.png', eigen)
            cv2.imwrite(new_dir_ + 'gt.png', gt)

if __name__ == '__main__':
    mr = merging()
    sbd_polar = "/home/park/PycharmProjects/PolarMask-master/work_dirs_polar_768_1x_r50_org_lr_down_sbd_24/test_vis/comparison/"
    sbd_eigen = "/home/park/PycharmProjects/Eigencontours/work_dirs_polar_768_1x_r50_360_36_sbd_24_rere_min/test_vis/comparison/"
    coco_polar = "/home/park/PycharmProjects/PolarMask-master/work_dirs_polar_768_1x_r50_org_lr_down/test_vis/comparison/"
    coco_eigen = "/home/park/PycharmProjects/Eigencontours/work_dirs_polar_768_1x_r50_360_36_lr_down/test_vis/comparison/"
    new_sbd = "/home/park/PycharmProjects/Eigencontours/comparison_sbd/"
    new_coco = "/home/park/PycharmProjects/Eigencontours/comparison_coco/"
    if not os.path.isdir(new_sbd):
        os.makedirs(new_sbd)
    if not os.path.isdir(new_coco):
        os.makedirs(new_coco)

    # mr.vis(sbd_eigen, sbd_polar, new_sbd)
    # mr.vis(coco_eigen, coco_polar, new_coco)

    new_save = "/home/park/PycharmProjects/Eigencontours/comparison_save_/"
    if not os.path.isdir(new_save):
        os.makedirs(new_save)

    # new_sbd_list = ['2011_000312_1', '2011_000575_0', '2011_001260_0', '2011_002158_0', '2011_003019_0',
    #                 '2010_004857_0', '2010_004654_0', '2010_004629_0', '2010_003933_0', '2010_003912_0', '2010_003468_0', '2010_003219_0', '2010_001892_0']
    # mr.save(sbd_eigen, sbd_polar, new_sbd_list, new_save)
    #
    # new_coco_list = ['000000570169_0', '000000523811_0', '000000471869_0', '000000464522_0', '000000451150_0', '000000402992_7', '000000379453_0', '000000378605_0', '000000375763_0', '000000363207_2',
    #                  '000000297343_0', '000000279774_2', '000000252701_1', '000000212559_0', '000000188439_0', '000000177015_0', '000000132408_0', '000000127955_1', '000000127517_1', '000000076417_0', '000000046804_0', ]
    new_coco_list = ["000000000885_1", "000000001000_2", "000000001000_3", "000000011760_2", "000000038210_0", "000000046804_0", "000000068387_1", "000000080671_0", "000000083113_0", "000000100624_1", "000000104619_0", "000000052996_2", "000000100723_4"]
    mr.save(coco_eigen, coco_polar, new_coco_list, new_save)
