import cv2
import math
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from collections import namedtuple
from libs.utils import *

Corner = namedtuple('Corner', 'x1 y1 x2 y2')
BBox = Corner
Center = namedtuple('Center', 'x y w h')

class Dataset_coco(Dataset):
    def __init__(self, cfg, datalist='train'):
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width

        truth = load_pickle("data/{}_seg".format(datalist))
        self.truth = truth

        truth_bb = load_pickle("data/{}_bb".format(datalist))
        self.truth_bb = truth_bb

        self.datalist = sorted(list(self.truth_bb.keys()))
        # self.datalist = load_pickle('datalist')

        # image transform
        self.transform = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
        self.new_shape = (cfg.height, cfg.width)

        self.imw, self.imh = self.cfg.crop_w, self.cfg.crop_h
        self.color_grey = (114, 114, 114)

        self.exemplar_size = 127
        self.search_size = 511
        self.context_amount = 0.5

    def transform(self, img, interpolation):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = cv2.resize(img, (self.width, self.height), interpolation=interpolation)
        return img

    def get_image(self, idx):
        img = cv2.imread(self.cfg.img_dir + self.datalist[idx] + '.jpg')[:, :, [2, 1, 0]]
        self.h, self.w, _ = img.shape
        # img = self.transform(img)
        # img = self.normalize(img)

        return img

    def get_label(self, idx):
        self.seg_label = self.truth[self.datalist[idx]]
        self.bb_label = self.truth_bb[self.datalist[idx]]

        cen_label = []
        id_label = []
        bbox_label = []
        poly_label = []

        for i in range(len(self.seg_label)):
            id_label.append(int(self.seg_label[i][0]))
            poly_pts = np.array(self.seg_label[i][1:]).reshape(-1, 2)
            poly_label.append(poly_pts)
            bb_pts = np.array(self.bb_label[i][:-1])
            cen = np.array([(bb_pts[0] + bb_pts[2])/2,(bb_pts[1] + bb_pts[3])/2])
            cen_label.append(cen.reshape(-1, 2))

        for j in range(len(self.bb_label)):
            bb_pts = np.array(self.bb_label[j][:-1])
            bbox = np.array([bb_pts[0], bb_pts[1], bb_pts[2]-bb_pts[0], bb_pts[3]-bb_pts[1]])
            bbox_label.append(bbox)

        return poly_label, cen_label, bbox_label, id_label

    def make_polygon_mask(self, label):
        img = Image.new("L", (self.w, self.h))
        ImageDraw.Draw(img).polygon(np.round(label).astype(np.float32), fill=1, outline=True)
        mask = np.array(img).astype(np.uint8)

        return mask

    def get_cropped_data(self, img, label, cen, bbox, id):
        outs = []

        for i in range(len(label)):
            out = {}
            crop_im, h, w = self.sample_target_SE(img, bbox[i], search_area_factor=1.0, out_h=self.cfg.crop_h, out_w=self.cfg.crop_w)
            crop_im = torch.from_numpy(crop_im).permute(2, 0, 1).type(torch.float)[[2, 1, 0], :, :] / 255
            crop_im_norm = self.normalize(crop_im)

            mask = self.make_polygon_mask(label[i])
            mask, h, w = self.sample_target_SE(to_3D_np(mask), bbox[i], search_area_factor=1.0, out_h=self.cfg.crop_h, out_w=self.cfg.crop_w)

            out['cropped_img_rgb'] = crop_im
            out['cropped_img'] = crop_im_norm
            out['seg_mask'] = mask[:, :, 0]
            out['crop_h'], out['crop_w'] = self.cfg.crop_h, self.cfg.crop_w
            out['bbox_center'] = np.array([(self.cfg.crop_w - 1) / 2, (self.cfg.crop_h - 1) / 2]).astype(np.float32)
            out['bbox'] = np.array([0, 0, self.cfg.crop_w - 1, self.cfg.crop_h - 1]).astype(np.float32)
            out['id'] = id[i]
            outs.append(out)

        return outs


    def __getitem__(self, idx):
        img = self.get_image(idx)
        label, cen, bbox, id = self.get_label(idx)
        outputs = self.get_cropped_data(img, label, cen, bbox, id)
        return {'output' : outputs,
                'img_name': self.datalist[idx]}

    def __len__(self):
        return len(self.datalist)

    def sample_target_SE(self, im, target_bb, search_area_factor=2.0, out_h=256, out_w=256, mode=0):
        """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

        args:
            im - cv image
            target_bb - target box [x, y, w, h]
            search_area_factor - Ratio of crop size to target size
            output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

        returns:
            cv image - extracted crop
            float - the factor by which the crop has been resized to make the crop size equal output_size
        """

        x, y, w, h = target_bb

        # Crop image
        ws = math.ceil(search_area_factor * w)
        hs = math.ceil(search_area_factor * h)

        if ws < 1 or hs < 1:
            im_crop_padded = cv2.resize(im, (out_w, out_h))
            if len(im_crop_padded.shape) == 2:
                im_crop_padded = im_crop_padded[..., np.newaxis]
            return im_crop_padded, 1.0, 1.0

        x1 = int(round(x + 0.5 * w - ws * 0.5))
        x2 = x1 + ws

        y1 = int(round(y + 0.5 * h - hs * 0.5))
        y2 = y1 + hs

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - im.shape[1] + 1, 0)

        y1_pad = max(0, -y1)
        y2_pad = max(y2 - im.shape[0] + 1, 0)

        # Crop target
        im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

        # Pad
        im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, mode)

        if out_h is not None:
            w_rsz_f = out_w / ws
            h_rsz_f = out_h / hs
            im_crop_padded_rsz = cv2.resize(im_crop_padded, (out_w, out_h))
            if len(im_crop_padded_rsz.shape) == 2:
                im_crop_padded_rsz = im_crop_padded_rsz[..., np.newaxis]
            return im_crop_padded_rsz, h_rsz_f, w_rsz_f
        else:
            return im_crop_padded, 1.0, 1.0
