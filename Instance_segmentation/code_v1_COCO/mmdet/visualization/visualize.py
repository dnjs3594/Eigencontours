import random

import cv2
import math

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageDraw
from mmdet.visualization.utils import *

class Visualize_cv(object):

    def __init__(self, save_dir = '../output_SBD'):

        self.save_dir = save_dir
        self.output_sz = 416
        self.width = self.output_sz
        self.height = self.output_sz
        self.img_scales = [768, 1280]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))

        # self.pickle_path = "../../data/vocsbdche/pickle_SBD"
        # self.norm = load_pickle(os.path.join(self.pickle_path, "offset_distribution"))
        # self.mean_c = self.norm['mean']
        # self.std_c = np.sqrt(self.norm['var'])

        self.line = np.zeros((self.height, 3, 3), dtype=np.uint8)
        self.line[:, :, :] = 255

        self.show = {}

        # result
        self.top_k = 36
        # self.node_num = cfg.node_num
        # self.thresd = cfg.seg_thresd

        self.cen = np.array([(self.width - 1) / 2, (self.height - 1) / 2])
        self.sft = torch.nn.Softmax(dim=0)

        # self.r_coord = cfg.r_coord
        # self.r_coord_x = cfg.r_coord_x
        # self.r_coord_y = cfg.r_coord_y
        # self.r_coord_xy = cfg.r_coord_xy
        #
        # # candidates
        # self.dist = math.sqrt(self.img_scales[0]*self.img_scales[0]/4 + self.img_scales[1]*self.img_scales[1]/4) / 10
        # self.U = load_pickle("/home/park/PycharmProjects/pre_data/U/U_iou").cpu()

    def update_org_image(self, path, name='org_img'):
        img = cv2.imread(path)
        self.show[name] = img

    def update_image(self, img, name='img'):
        img = to_np(img.permute(1, 2, 0))
        img = np.uint8(img * 255)[:, :, [2, 1, 0]]
        # img = np.uint8((img * self.std + self.mean) * 255)[:, :, [2, 1, 0]]
        self.show[name] = img

    def update_label(self, label, name='label'):
        label = to_np(label)
        label = np.repeat(np.expand_dims(np.uint8(label != 0) * 255, axis=2), 3, 2)
        self.show[name] = label

    def update_data(self, data, name=None):
        self.show[name] = data

    def update_image_name(self, img_name):
        self.show['img_name'] = img_name

    def multi_edge_map_to_rgb_image(self, data):
        data = torch.max(data[1:], dim=0, keepdim=True)[0]
        data = np.repeat(np.uint8(to_np2(data.permute(1, 2, 0) * 255)), 3, 2)
        data = cv2.resize(data, (self.width, self.height))
        return data

    def edge_map_to_rgb_image(self, data):
        data = np.repeat(np.uint8(to_np2(data.permute(1, 2, 0) * 255)), 3, 2)
        data = cv2.resize(data, (self.width, self.height))
        return data

    def draw_text(self, pred, label, name, ref_name='img', color=(255, 0, 0)):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        cv2.rectangle(img, (1, 1), (250, 120), color, 1)
        cv2.putText(img, 'pred : ' + str(pred), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, 'label : ' + str(label), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        self.show[name] = img

    def draw_polyline_cv(self, data, name, ref_name='img', color=(255, 0, 0), s=2):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        pts = np.int32(data).reshape((-1, 1, 2))
        img = cv2.polylines(img, [pts], True, color, s,
                            lineType=cv2.LINE_AA)
        self.show[name] = img


    def draw_lines_cv(self, data, name, ref_name='img', color=(255, 0, 0), s=2):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        for i in range(data.shape[0]):
            pts = data[i]
            pt_1 = (pts[0], pts[1])
            pt_2 = (pts[2], pts[3])
            img = cv2.line(img, pt_1, pt_2, color, s)

        self.show[name] = img

    def draw_lane_points_cv(self, data_x, data_y, name, ref_name='org_img', color=(0, 255, 0), s=4):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        for i in range(len(data_x)):
            for j in range(data_x[i].shape[0]):
                pts = (int(data_x[i][j]), int(data_y[i][j]))
                img = cv2.circle(img, pts, s, color, -1)

        self.show[name] = img

    def draw_mask_cv(self, data, name, ref_name='img', color=(255, 0, 0)):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        mask = (np.repeat(data[..., np.newaxis], 3, axis=2) * np.array(color).reshape(1, 1, -1)).astype(np.uint8)
        img += mask
        self.show[name] = img

    def draw_mask_cv_2(self, data, name, ref_name="img", color_code=2):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        img[:, :, color_code] = data*255 + (1-data)*img[:, :, color_code]
        self.show[name] = img

    def draw_mask_cv_3(self, data, name, ref_name='img', color=(0, 0, 255)):
        image = np.ascontiguousarray(np.copy(self.show[ref_name]))
        alpha = 0.5
        color = list(color)
        # for i in range(3):
        #     img[:, :, i] = data * color[i] * 0.5 + data * 255 * 0.5 + (1 - data) * img[:, :, i]
        for n, c in enumerate(color):
            image[:, :, n] = np.where(
                data == 1,
                image[:, :, n] * (1 - alpha) + alpha * c,
                image[:, :, n]
            )

        self.show[name] = image

    def display_saveimg(self, dir_name, file_name, list):
        # boundary line
        # if self.show[list[0]].shape[0] != self.line.shape[0]:
        #     self.line = np.zeros((self.show[list[0]].shape[0], 3, 3), dtype=np.uint8)
        #     self.line[:, :, :] = 255
        # disp = self.line

        disp = np.array([], dtype=np.uint8)

        for name in list:
            if disp.shape[0] == 0:
                disp = self.show[name]
                continue
            disp = np.concatenate((disp, self.show[name]), axis=1)

        mkdir(dir_name)
        cv2.imwrite(os.path.join(dir_name, file_name), disp)

    def generator_coeff_cen(self, center, coeff):
        output_sz = 16  # same as input resolution

        center = center[0].cpu().detach().numpy()
        coeff = coeff[0].cpu().detach().numpy()

        return center, coeff

    def make_shape_center(self, center, coeff):

        # start.record()
        # self.max_dist = 128 / 2 * math.sqrt(2)
        center, coeff = self.generator_coeff_cen(center, coeff)
        pred_coeff = coeff.reshape(-1, 1)
        cen = np.repeat(center.reshape(1, -1), self.node_num, 0)
        pred_r_coord = np.matmul(self.U, pred_coeff) * self.max_dist
        pred_xy = self.r_coord_xy * pred_r_coord
        polygon_pts_pred = cen + pred_xy
        img = Image.new("L", (256, 256))
        ImageDraw.Draw(img).polygon(polygon_pts_pred.astype(np.float32), fill=1, outline=True)
        mask_pred = np.array(img)
        # end.record()

        return mask_pred

    def display_for_train(self, batch, out, epoch, itr):

        n_of_batch = 0
        self.update_image(batch['test_images'][n_of_batch, 0], name='img')

        list = ['ref_img', 'img', 'pred_coeff', 'pred_mask', 'pred_corr', 'gt_coeff', 'gt_mask']

        for name in list:
            self.show[name] = self.show['img'].copy()
        self.update_image(batch['train_images'][n_of_batch, 0], name='ref_img')

        p_cm = self.make_shape_center(out['center'], out['coeff'])
        bbox_gt = batch['test_anno'].squeeze(1)
        bbox_gt[:, 2:] += bbox_gt[:, :2]
        bbox_gt_center = bbox_gt.view(-1, 2, 2).mean(dim=1)
        gt_cm = self.make_shape_center(bbox_gt_center, batch['test_coeff'].squeeze(1))

        p_sm = to_np2(out['mask'][n_of_batch, 0])
        p_corr = self.fusion_mask(p_sm, p_cm)
        p_sm = (p_sm > self.thresd).astype(np.uint8)
        gt_sm = to_np(batch['test_masks'][n_of_batch, 0, 0]).astype(np.uint8)

        self.show['pred_coeff'][:, :, 2] = p_cm * 255 + (1 - p_cm) * self.show['pred_coeff'][:, :, 2]
        self.show['pred_coeff'] = cv2.circle(self.show['pred_coeff'], (int(out['center'][n_of_batch][0].item()), int(out['center'][n_of_batch][1].item())), 3, (0, 255, 255), -1)

        self.show['pred_mask'][:, :, 2] = p_sm * 255 + (1 - p_sm) * self.show['pred_mask'][:, :, 2]
        self.show['pred_corr'][:, :, 2] = p_corr * 255 + (1 - p_corr) * self.show['pred_corr'][:, :, 2]

        self.show['gt_coeff'][:, :, 1] = gt_cm * 255 + (1 - gt_cm) * self.show['gt_coeff'][:, :, 1]
        self.show['gt_coeff'] = cv2.circle(self.show['gt_coeff'], (int(bbox_gt_center[0][0].item()), int(bbox_gt_center[0][1].item())), 3, (0, 255, 255), -1)
        self.show['gt_mask'][:, :, 1] = gt_sm * 255 + (1 - gt_sm) * self.show['gt_mask'][:, :, 1]

        # im_show = np.concatenate((img, pred_coeff, pred_mask, pred_corr, gt_coeff, gt_mask), axis=1)
        self.display_saveimg(dir_name=self.cfg.dir['out'] + 'train/display/',
                             file_name= '{}_{}.png'.format(str(epoch), str(itr)),
                             list=list)

    def fusion_mask(self, mask, shape):

        if mask.sum() < 10:
            return shape
        elif shape.sum() < 10:
            return mask
        mask_ = (mask > 0.65).astype(np.float)
        overlap = mask_ + shape
        uni = (overlap != 0).astype(np.uint8)
        inter = (overlap == 2).astype(np.uint8)

        # IoU = inter.sum() / uni.sum()

        prime = (mask * shape > 0.1).astype(np.float32)
        detail = (mask > 0.9).astype(np.float32)

        return (prime + detail != 0).astype(np.float32)

    def display_for_test_init(self, data, video, f):

        list = ['img', 'gt']

        self.show['img'] = data[0][:, :, [2, 1, 0]]
        self.show['gt'] = self.show['img'].copy()

        self.show['gt'][:, :, 2] = data[1] * 255 + (1 - data[1]) * self.show['gt'][:, :, 2]
        self.show['gt'][:, :, 1] = data[1] * 255 + (1 - data[1]) * self.show['gt'][:, :, 1]

        self.display_saveimg(dir_name=os.path.join(self.cfg.dir['out'], 'val', 'org', video),
                             file_name='{:08d}.png'.format(f),
                             list=list)

        self.show['img'] = data[2][:, :, [2, 1, 0]]
        self.show['gt'] = self.show['img'].copy()

        data[3] = data[3][:, :, 0]
        self.show['gt'][:, :, 2] = data[3] * 255 + (1 - data[3]) * self.show['gt'][:, :, 2]
        self.show['gt'][:, :, 1] = data[3] * 255 + (1 - data[3]) * self.show['gt'][:, :, 1]

        self.display_saveimg(dir_name=os.path.join(self.cfg.dir['out'], 'val', 'crop', video),
                             file_name='{:08d}.png'.format(f),
                             list=list)

    def display_for_test(self, data, video, f, refine=False):

        list = ['img', 'pred', 'gt']

        self.show['img'] = data[0][:, :, [2, 1, 0]]
        self.show['pred'] = self.show['img'].copy()
        self.show['gt'] = self.show['img'].copy()

        self.show['pred'][:, :, 2] = data[1] * 255 + (1 - data[1]) * self.show['pred'][:, :, 2]

        self.show['gt'][:, :, 2] = data[2] * 255 + (1 - data[2]) * self.show['gt'][:, :, 2]
        self.show['gt'][:, :, 1] = data[2] * 255 + (1 - data[2]) * self.show['gt'][:, :, 1]

        if refine == True:
            self.display_saveimg(dir_name=os.path.join(self.cfg.dir['out'], 'val', 'org_refine', video),
                                 file_name='{:08d}.png'.format(f),
                                 list=list)
        else:
            self.display_saveimg(dir_name=os.path.join(self.cfg.dir['out'], 'val', 'org', video),
                                 file_name='{:08d}.png'.format(f),
                                 list=list)


        list = ['img', 'seg_prob', 'seg', 'shape', 'corr', 'gt']

        self.show['img'] = data[3][:, :, [2, 1, 0]]

        for name in list:
            if name == 'seg_prob':
                continue
            self.show[name] = self.show['img'].copy()

        self.show['seg_prob'] = np.repeat((data[4]*255).astype(np.uint8)[:, :, np.newaxis], 3, 2)

        seg_mask = (data[4] > self.cfg.seg_thresd).astype(np.uint8)
        self.show['seg'][:, :, 1] = seg_mask * 255 + (1 - seg_mask) * self.show['seg'][:, :, 1]

        self.show['shape'][:, :, 2] = data[5] * 255 + (1 - data[5]) * self.show['shape'][:, :, 2]

        self.show['corr'][:, :, 2] = data[6] * 255 + (1 - data[6]) * self.show['corr'][:, :, 2]

        data[7] = data[7][:, :, 0]
        self.show['gt'][:, :, 2] = data[7] * 255 + (1 - data[7]) * self.show['gt'][:, :, 2]
        self.show['gt'][:, :, 1] = data[7] * 255 + (1 - data[7]) * self.show['gt'][:, :, 1]

        if refine == True:
            self.display_saveimg(dir_name=os.path.join(self.cfg.dir['out'], 'val', 'crop_refine', video),
                                 file_name='{:08d}.png'.format(f),
                                 list=list)
        else:
            self.display_saveimg(dir_name=os.path.join(self.cfg.dir['out'], 'val', 'crop', video),
                                 file_name='{:08d}.png'.format(f),
                                 list=list)

    def plot_box(self, img, pred, target, img_name):

        self.update_image(img[0])

        list = ['img', 'pred', 'gt']
        self.show['pred'] = self.show['img'].copy()
        self.show['gt'] = self.show['img'].copy()

        if pred[0] is not None:
            pred_vis = pred[0][pred[0][:, 4] > 0.2]
            if pred_vis.shape[0] != 0:
                for i in range(pred_vis.shape[0]):
                    bbox = pred_vis[i, : 4].numpy()
                    c1 = (int(bbox[0]), int(bbox[1]))
                    c2 = (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(self.show['pred'], c1, c2, (255, 0, 0), 2, cv2.LINE_AA)

        if target.shape[0] != 0:
            for j in range(target.shape[0]):
                bbox = target[j][2:6].numpy()
                c1 = (int(bbox[0]), int(bbox[1]))
                c2 = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(self.show['gt'], c1, c2, (0, 0, 255), 2, cv2.LINE_AA)

        self.display_saveimg(dir_name=os.path.join(self.save_dir, 'val', 'bbox'),
                             file_name= img_name,
                             list=list)
    def random(self):
        list = []
        for i in range(3):
            list.append(random.randrange(0, 256, 1))

        return tuple(list)

    def plot_all(self, img, pred, target, img_name, scale_param):

        self.update_image(img[0])

        list = ['img', 'pred_bbox', 'gt_bbox', 'pred_shape', 'gt_shape']
        self.show['pred_bbox'] = self.show['img'].copy()
        self.show['gt_bbox'] = self.show['img'].copy()
        self.show['pred_shape'] = self.show['img'].copy()
        self.show['gt_shape'] = self.show['img'].copy()

        if pred[0] is not None:
            pred_vis = pred[0][pred[0][:, 4] > 0.2]
            if pred_vis.shape[0] != 0:
                for i in range(pred_vis.shape[0]):
                    bbox = pred_vis[i, : 4].numpy()
                    c1 = (int(bbox[0]), int(bbox[1]))
                    c2 = (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(self.show['pred_bbox'], c1, c2, (255, 0, 0), 2, cv2.LINE_AA)

                    center = pred_vis[i, 7:9].numpy()
                    center_xs = center[0].reshape(-1, 1)  # N,1
                    center_ys = center[1].reshape(-1, 1)
                    coefs = pred_vis[i, 9:].numpy()
                    coefs = coefs * self.std_c + self.mean_c
                    rs = np.matmul(self.U, coefs) * self.max_dist * scale_param[0]
                    rs = rs.astype(np.float32)[np.newaxis, :] # N, 360
                    theta_list = np.arange(359, -1, -1).reshape(1, 360)  # 1, 360 1, 360
                    theta_list = theta_list.repeat(int(rs.shape[0]), axis=0).astype(np.float32)  # N,360
                    x, y = cv.polarToCart(rs, theta_list, angleInDegrees=True)  # N,360    N,360
                    x = x + center_xs.astype(np.float32)  # N.360
                    y = y + center_ys.astype(np.float32)  # N,360

                    x = np.clip(x, bbox[0], bbox[2]).reshape(-1, 360, 1)  # N,360,1
                    y = np.clip(y, bbox[1], bbox[3]).reshape(-1, 360, 1)  # N,360,1
                    polygons = np.concatenate((x, y), axis=2)[0]  # N,360,2

                    im = Image.new("L", (416, 416))
                    ImageDraw.Draw(im).polygon(polygons, fill=1, outline=True)
                    mask = np.array(im)
                    rand = self.random()
                    self.draw_mask_cv_3(mask, name='pred_shape', ref_name='pred_shape', color=rand)
                    # self.draw_mask_cv_3(mask, name='pred_shape', ref_name='pred_shape', color=(0, 0, 255))
                    self.draw_polyline_cv(polygons, name='pred_shape', ref_name='pred_shape', color=rand, s=1)

        if target.shape[0] != 0:
            for j in range(target.shape[0]):
                bbox = target[j][2:6].numpy()
                c1 = (int(bbox[0]), int(bbox[1]))
                c2 = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(self.show['gt_bbox'], c1, c2, (0, 0, 255), 2, cv2.LINE_AA)

                target_polygons_x = target[j][9:-360].numpy()
                target_polygons_y = target[j][-360:].numpy()
                target_polygons = np.stack((target_polygons_x, target_polygons_y), axis=1)
                im = Image.new("L", (416, 416))
                ImageDraw.Draw(im).polygon(target_polygons, fill=1, outline=True)
                mask = np.array(im)
                self.draw_mask_cv_3(mask, name='gt_shape', ref_name='gt_shape', color=(0, 255, 0))
                self.draw_polyline_cv(target_polygons, name='gt_shape', ref_name='gt_shape', color=(0, 255, 0), s=2)

        self.display_saveimg(dir_name=os.path.join(self.save_dir, 'val', 'all'),
                             file_name= img_name,
                             list=list)

    def plot_all_cheby(self, img, pred, target, img_name, mode='val'):

        self.update_image(img[0])

        list = ['img', 'pred_bbox', 'gt_bbox', 'pred_shape', 'gt_shape']
        self.show['pred_bbox'] = self.show['img'].copy()
        self.show['gt_bbox'] = self.show['img'].copy()
        self.show['pred_shape'] = self.show['img'].copy()
        self.show['gt_shape'] = self.show['img'].copy()

        if pred[0] is not None:
            pred_vis = pred[0][pred[0][:, 4] > 0.2]
            if pred_vis.shape[0] != 0:
                for i in range(pred_vis.shape[0]):
                    bbox = pred_vis[i, : 4].numpy()
                    c1 = (int(bbox[0]), int(bbox[1]))
                    c2 = (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(self.show['pred_bbox'], c1, c2, (255, 0, 0), 2, cv2.LINE_AA)

                    center = pred_vis[i, 7:9].numpy()
                    coefs = pred_vis[i, 9:].numpy()
                    bboxs_x1 = bbox[0].reshape(-1, 1)  # N,1
                    bboxs_x2 = bbox[2].reshape(-1, 1)  # N,1
                    bboxs_y1 = bbox[1].reshape(-1, 1)  # N,1
                    bboxs_y2 = bbox[3].reshape(-1, 1)  # N,1
                    bboxsw = np.abs(bboxs_x2 - bboxs_x1)  # N,1
                    bboxsh = np.abs(bboxs_y2 - bboxs_y1)  # N,1
                    relative_lens = np.sqrt(bboxsw * bboxsw + bboxsh * bboxsh)  # N,1
                    center_xs = center[0].reshape(-1, 1)  # N,1
                    center_ys = center[1].reshape(-1, 1)  # N,1
                    rs = cheby(coefs) * relative_lens  # N, 360
                    rs = rs.astype(np.float32)  # N, 360
                    theta_list = np.arange(359, -1, -1).reshape(1, 360)  # 1, 360
                    theta_list = theta_list.repeat(int(rs.shape[0]), axis=0).astype(np.float32)  # N,360
                    x, y = cv.polarToCart(rs, theta_list, angleInDegrees=True)  # N,360    N,360
                    x = x + center_xs.astype(np.float32)  # N.360
                    y = y + center_ys.astype(np.float32)  # N,360

                    x = np.clip(x, bboxs_x1, bboxs_x2).reshape(-1, 360, 1)  # N,360,1
                    y = np.clip(y, bboxs_y1, bboxs_y2).reshape(-1, 360, 1)  # N,360,1
                    polygons = np.concatenate((x, y), axis=-1)[0]  # N,360,2

                    self.draw_polyline_cv(polygons, name='pred_shape', ref_name='pred_shape', color=(255, 0, 0), s=2)

        if target.shape[0] != 0:
            for j in range(target.shape[0]):
                bbox = target[j][2:6].numpy()
                c1 = (int(bbox[0]), int(bbox[1]))
                c2 = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(self.show['gt_bbox'], c1, c2, (0, 0, 255), 2, cv2.LINE_AA)

                center = target[j, 6:8].numpy()
                coefs = target[j, 8:-3].numpy()
                bboxs_x1 = bbox[0].reshape(-1, 1)  # N,1
                bboxs_x2 = bbox[2].reshape(-1, 1)  # N,1
                bboxs_y1 = bbox[1].reshape(-1, 1)  # N,1
                bboxs_y2 = bbox[3].reshape(-1, 1)  # N,1
                bboxsw = np.abs(bboxs_x2 - bboxs_x1)  # N,1
                bboxsh = np.abs(bboxs_y2 - bboxs_y1)  # N,1
                relative_lens = np.sqrt(bboxsw * bboxsw + bboxsh * bboxsh)  # N,1
                center_xs = center[0].reshape(-1, 1)  # N,1
                center_ys = center[1].reshape(-1, 1)  # N,1
                rs = cheby(coefs) * relative_lens  # N, 360
                rs = rs.astype(np.float32)  # N, 360
                theta_list = np.arange(359, -1, -1).reshape(1, 360)  # 1, 360
                theta_list = theta_list.repeat(int(rs.shape[0]), axis=0).astype(np.float32)  # N,360
                x, y = cv.polarToCart(rs, theta_list, angleInDegrees=True)  # N,360    N,360
                x = x + center_xs.astype(np.float32)  # N.360
                y = y + center_ys.astype(np.float32)  # N,360

                x = np.clip(x, bboxs_x1, bboxs_x2).reshape(-1, 360, 1)  # N,360,1
                y = np.clip(y, bboxs_y1, bboxs_y2).reshape(-1, 360, 1)  # N,360,1
                polygons = np.concatenate((x, y), axis=-1)[0]  # N,360,2

                self.draw_polyline_cv(polygons, name='gt_shape', ref_name='gt_shape', color=(0, 0, 255), s=2)

        self.display_saveimg(dir_name=os.path.join(self.save_dir, mode, 'all_cheby'),
                             file_name=img_name,
                             list=list)


    def plot_all_cheby_train(self, img, pred, target, epoch, batch_idx, img_name, scale_param, mode='val'):

        self.update_image(img[0])

        list = ['img', 'pred_bbox', 'gt_bbox', 'pred_shape', 'gt_shape']
        self.show['pred_bbox'] = self.show['img'].copy()
        self.show['gt_bbox'] = self.show['img'].copy()
        self.show['pred_shape'] = self.show['img'].copy()
        self.show['gt_shape'] = self.show['img'].copy()

        if pred[0] is not None:
            pred_vis = pred[0][pred[0][:, 4] > 0.2]
            if pred_vis.shape[0] != 0:
                for i in range(pred_vis.shape[0]):
                    bbox = pred_vis[i, : 4].numpy()
                    c1 = (int(bbox[0]), int(bbox[1]))
                    c2 = (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(self.show['pred_bbox'], c1, c2, (255, 0, 0), 2, cv2.LINE_AA)

                    center = pred_vis[i, 7:9].numpy()
                    center_xs = center[0].reshape(-1, 1)  # N,1
                    center_ys = center[1].reshape(-1, 1)
                    coefs = pred_vis[i, 9:].numpy()
                    coefs = coefs * self.std_c + self.mean_c
                    rs = np.matmul(self.U, coefs) * self.max_dist * scale_param[0]
                    rs = rs.astype(np.float32)[np.newaxis, :] # N, 360
                    theta_list = np.arange(359, -1, -1).reshape(1, 360)  # 1, 360 1, 360
                    theta_list = theta_list.repeat(int(rs.shape[0]), axis=0).astype(np.float32)  # N,360
                    x, y = cv.polarToCart(rs, theta_list, angleInDegrees=True)  # N,360    N,360
                    x = x + center_xs.astype(np.float32)  # N.360
                    y = y + center_ys.astype(np.float32)  # N,360

                    # x = np.clip(x, bboxs_x1, bboxs_x2).reshape(-1, 360, 1)  # N,360,1
                    # y = np.clip(y, bboxs_y1, bboxs_y2).reshape(-1, 360, 1)  # N,360,1
                    polygons = np.stack((x, y), axis=2)[0]  # N,360,2

                    self.draw_polyline_cv(polygons, name='pred_shape', ref_name='pred_shape', color=(255, 0, 0), s=2)

        target = target[target[:, 0] == 0].cpu()
        if target.shape[0] != 0:
            for j in range(target.shape[0]):
                bbox = target[j][2:6].numpy()
                c1 = (int(bbox[0]), int(bbox[1]))
                c2 = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(self.show['gt_bbox'], c1, c2, (0, 0, 255), 2, cv2.LINE_AA)

                center = target[j, 6:8].numpy()
                coefs = target[j, 8:-3].numpy()
                center_xs = center[0].reshape(-1, 1)  # N,1
                center_ys = center[1].reshape(-1, 1)
                # bboxs_x1 = bbox[0].reshape(-1, 1)  # N,1
                # bboxs_x2 = bbox[2].reshape(-1, 1)  # N,1
                # bboxs_y1 = bbox[1].reshape(-1, 1)  # N,1
                # bboxs_y2 = bbox[3].reshape(-1, 1)  # N,1
                # bboxsw = np.abs(bboxs_x2 - bboxs_x1)  # N,1
                # bboxsh = np.abs(bboxs_y2 - bboxs_y1)  # N,1
                # relative_lens = np.sqrt(bboxsw * bboxsw + bboxsh * bboxsh)  # N,1
                coefs = coefs * self.std_c + self.mean_c
                rs = np.matmul(self.U, coefs) * self.max_dist * scale_param[0]
                rs = rs.astype(np.float32)[np.newaxis, :]  # N, 360
                theta_list = np.arange(359, -1, -1).reshape(1, 360)  # 1, 360
                theta_list = theta_list.repeat(int(rs.shape[0]), axis=0).astype(np.float32)  # N,360
                x, y = cv.polarToCart(rs, theta_list, angleInDegrees=True)  # N,360    N,360
                x = x + center_xs.astype(np.float32)  # N.360
                y = y + center_ys.astype(np.float32)  # N,360

                # x = np.clip(x, bboxs_x1, bboxs_x2).reshape(-1, 360, 1)  # N,360,1
                # y = np.clip(y, bboxs_y1, bboxs_y2).reshape(-1, 360, 1)  # N,360,1
                polygons = np.stack((x, y), axis=2)[0]  # N,360,2

                self.draw_polyline_cv(polygons, name='gt_shape', ref_name='gt_shape', color=(0, 0, 255), s=2)

        self.display_saveimg(dir_name=os.path.join(self.save_dir, mode, 'all_cheby'),
                             file_name= str(epoch) + '_' + str(batch_idx) + '_' + img_name,
                             list=list)

    def plot_train(self, pred, label, epoch=0, idx=1):

        list = ['img', 'pred', 'pred_c', 'gt']
        self.show['img'] = label[0]
        self.show['pred'] = pred[0]
        self.show['pred_c'] = label[1]
        self.show['gt'] = label[2]

        img_name = "train_{}_{}.jpg".format(str(epoch), str(idx))

        self.display_saveimg(dir_name=self.save_dir,
                             file_name=img_name,
                             list=list)