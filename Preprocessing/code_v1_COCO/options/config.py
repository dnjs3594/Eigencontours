import os
import math

class Config(object):
    def __init__(self):
        self.head_path = os.path.dirname(os.getcwd()) + '/'

        # dataset dir
        self.dataset = 'coco'  # ['coco', 'cityscape']
        self.datalist = 'train'  # ['train', 'test', 'val']
        self.mode = "encoding"  # ['encoding', 'svd', 'convert']

        self.img_dir = self.head_path + 'data/coco/{}2017/'.format(self.datalist)
        self.label_path_seg = "{}_seg.txt".format(self.datalist)
        self.label_path_bb = "{}_bb.txt".format(self.datalist)

        self.node_num = 360

        self.output_dir = self.head_path + 'output_' + self.dataset + '_' + self.datalist + \
                          '_v1_node_{}/'.format(self.node_num)

        # other setting
        self.process_mode = 'ese_ori'

        # dataloader
        self.gpu_id = "0"
        self.num_workers = 4
        self.batch_size = 1

        # constant
        self.height = 416
        self.width = 416
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.crop_w, self.crop_h = 256, 256
        self.max_dist = math.sqrt(self.width / 2 * self.width / 2 + self.height / 2 * self.height / 2)

        self.dim = 36
  
        self.thresd_iou = 0.85

        # visualization
        self.display = False

        # save
        self.save_pickle = True
