import logging
from abc import ABCMeta, abstractmethod
import os
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch.nn as nn
import torch

from mmdet.core import auto_fp16, get_classes, tensor2imgs

import math
import pickle

from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw
import cv2
def load_pickle(file_path):
    with open(file_path + '.pickle', 'rb') as f:
        data = pickle.load(f)

    return data

class BaseDetector(nn.Module):
    """Base class for detectors"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False
        self.angles = torch.range(0, 359, 1) / 180 * math.pi
        # self.U = load_pickle("/data/CVPR2022/project/Preprocessing/E02/output_coco_train_v1_node_360_polarmask_not_flip/pickle/U").cpu()

        # self.U = load_pickle("/home/park/PycharmProjects/pre_data/U/U_iou").cpu()
        # self.dist = math.sqrt(self.img_scales[0][0]*self.img_scales[0][0]/4 + self.img_scales[0][1]*self.img_scales[0][1]/4) / 10
        self.U = load_pickle("data_pickle/U_iou").cpu()
    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    def show_result(self,
                    data,
                    result,
                    img_norm_cfg,
                    dataset=None,
                    score_thr=0.3,
                    outdir='./',
                    idx=0):
        if isinstance(result, tuple):
            bbox_result, segm_result, _ = result
        else:
            bbox_result, segm_result, _ = result, None

        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        org_img = imgs[0].copy()
        assert len(imgs) == len(img_metas)

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))
        result_images = []
        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            bboxes = np.vstack(bbox_result)
            # draw segmentation masks
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    color_mask = np.random.randint(
                        0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
                    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
                    contour = contours[np.argmax(cnt_area)]  # use max area polygon
                    pred_polygon = contour.reshape(-1, 1, 2)
                    img_show = cv2.polylines(img_show, [pred_polygon], True, (color_mask[0][0].item(), color_mask[0][1].item(), color_mask[0][2].item()), 2, lineType=cv2.LINE_AA)
            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            result_image = mmcv.imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                class_names=class_names,
                score_thr=score_thr,
                out_file=os.path.join(outdir, "test_{}.jpg".format(str(idx))))
            result_images.append(result_image)

        return result_images, org_img

    def show_result_train(self,
                    data,
                    result,
                    img_norm_cfg,
                    dataset=None,
                    score_thr=0.3,
                    outdir='./',
                    epoch=0,
                    idx=0):
        if isinstance(result, tuple):
            bbox_result, segm_result, _ = result
        else:
            bbox_result, segm_result, _  = result, None

        img_tensor = data['img'].data[0][:1]
        img_metas = [data['img_meta'].data[0][0]]
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))

        result_images = []
        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            bboxes = np.vstack(bbox_result)
            # draw segmentation masks
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    color_mask = np.random.randint(
                        0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            result_image = mmcv.imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                class_names=class_names,
                score_thr=score_thr,
                out_file=None)
            result_images.append(result_image)

        return result_images

    def show_result_gt(self,
                          data,
                          result,
                          img_norm_cfg,
                          dataset=None,
                          score_thr=0.3,
                          outdir='./',
                          epoch=0,
                          idx=0,
                          U=None,
                          dist=None):
        if isinstance(result, tuple):
            bbox_result, segm_result, _ = result
        else:
            bbox_result, segm_result, _ = result, None

        img_tensor = data['img'].data[0][:1]
        img_metas = [data['img_meta'].data[0][0]]
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)
        h, w, _ =  imgs[0].shape
        org_img = imgs[0].copy()
        tmp_img = imgs[0].copy()
        cs = data['gt_cs'].data[0][0]
        segms = data['gt_masks'].data[0][0]
        bboxes = data['gt_bboxes'].data[0][0].numpy()
        scores = np.ones((bboxes.shape[0], 1)).astype(np.float32)
        bboxes = np.concatenate((bboxes, scores), axis=1)
        labels = data['gt_labels'].data[0][0].numpy().astype(np.int32) - 1
        n, _ = bboxes.shape
        # center = torch.tensor(bboxes[:, :-1].reshape(n, -1, 2).mean(axis=1))

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))

        result_images = []
        result_images.append(org_img)

        center = cs[:, :2]
        num_points = center.shape[0]
        points = center[:, :, None].repeat(1, 1, 360)
        c_x, c_y = points[:, 0], points[:, 1]

        sin = torch.sin(self.angles)
        cos = torch.cos(self.angles)
        sin = sin[None, :].repeat(num_points, 1)
        cos = cos[None, :].repeat(num_points, 1)

        self.dist = math.sqrt(w * w / 4 + h * h / 4) / 10
        distances = (torch.matmul(self.U[:, :36], cs[:, 2:].T) * self.dist).T

        x = distances * sin + c_x
        y = distances * cos + c_y

        x = x.clamp(min=0, max=w - 1)
        y = y.clamp(min=0, max=h - 1)

        res = torch.cat([x[:, None, :], y[:, None, :]], dim=1)

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = tmp_img[:h, :w, :]

            n = res.shape[0]
            # draw segmentation masks
            img = np.ascontiguousarray(np.copy(img_show))
            for i in range(n):
                color_mask = np.random.randint(
                    0, 256, (1, 3), dtype=np.uint8)
                pts = res[i].T.numpy()
                # img = Image.new("L", (w, h))
                # ImageDraw.Draw(img).polygon(pts, fill=1, outline=True)
                # mask = np.array(img).astype(np.bool)
                # im_mask = np.zeros((h, w), dtype=np.uint8)
                # try:
                for p in range(len(pts)):
                    pt = (int(pts[p][0]), int(pts[p][1]))
                    img = cv2.circle(img, pt, 2, (0, 0, 255), -1)
                # cen_pt = (int(center[i][0]), int(center[i][1]))
                # img = cv2.circle(img, cen_pt, 3, (0, 255, 255), -1)
                # img = Image.new("L", (w, h))
                # ImageDraw.Draw(img).polygon(pts[idx], fill=1, outline=True)
                # mask = np.array(img).astype(np.bool)
                # # mask = [pts[:, np.newaxis, :]]
                # # mask_a = cv2.drawContours(im_mask, mask, -1, 1, -1)
                # img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
                # except:
                #     continue
                # hull = ConvexHull(pts)
                # idx = hull.vertices
                # img = Image.new("L", (w, h))
                # ImageDraw.Draw(img).polygon(pts[idx], fill=1, outline=True)
                # mask = np.array(img).astype(np.bool)
                # # mask = [pts[:, np.newaxis, :]]
                # # mask_a = cv2.drawContours(im_mask, mask, -1, 1, -1)
                # img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5

            # draw bounding boxes
            result_image_c = mmcv.imshow_det_bboxes(
                img,
                bboxes,
                labels,
                class_names=class_names,
                score_thr=score_thr,
                out_file=None)
        result_images.append(result_image_c)

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            n = segms.shape[0]
            # draw segmentation masks
            for i in range(n):
                color_mask = np.random.randint(
                    0, 256, (1, 3), dtype=np.uint8)
                mask = segms[i].astype(np.bool)
                img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
            # draw bounding boxes
            result_image = mmcv.imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                class_names=class_names,
                score_thr=score_thr,
                out_file=None)
            result_images.append(result_image)
        return result_images