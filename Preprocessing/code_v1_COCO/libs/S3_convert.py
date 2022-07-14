import cv2
import math
from PIL import Image, ImageDraw
from tqdm import tqdm
from libs.utils import *

from davisinteractive.metrics import batched_f_measure

class Convert(object):

    def __init__(self, cfg, dict_DB):

        self.cfg = cfg
        self.dataloader = dict_DB['dataloader']
        self.visualize = dict_DB['visualize']
        self.height = cfg.height
        self.width = cfg.width
        self.size = np.float32([cfg.height, cfg.width])

        self.center = np.array([(self.width - 1) / 2, (self.height - 1) / 2])
        self.datalist = []

        self.r_coord = np.linspace(0, 360, self.cfg.node_num, endpoint=False)
        self.r_coord_x = np.cos(self.r_coord*2*math.pi/360).reshape(-1, 1)
        self.r_coord_y = ((-1) * np.sin(self.r_coord*2*math.pi/360)).reshape(-1, 1)
        self.r_coord_xy = np.concatenate((self.r_coord_x, self.r_coord_y), axis=1).astype(np.float32)

    def approximate_contour(self):
        out = self.data

        results = []
        r_coord = torch.FloatTensor([]).cuda()
        r_coord_id = []
        r_coord_cen = []
        check = np.full(len(out), True, dtype=np.bool)

        for i in range(len(out)):
            r_coord_tmp = out[i]['r']
            if len(r_coord_tmp) == 0:
                check[i] = False
            else:
                r_coord = torch.cat((r_coord, torch.tensor(r_coord_tmp).type(torch.float32).view(-1, 1).cuda()), dim=1)
                r_coord_id.append(out[i]['id_xyxy'][-1])
                r_coord_cen.append(np.array(out[i]['center']))

        dir_ = self.cfg.output_dir + '/display_approx/' + self.img_name
        mkdir(dir_)

        n = 0
        for i in range(len(out)):
            if check[i] == False:
                result = {}
                result['r'] = np.array([])
                result['c'] = np.array([])
                result['center'] = np.array([])
                result['id_xyxy'] = np.array([])
                results.append(result)
                continue

            U = self.U[:, :self.cfg.dim]
            U_t = U.clone().permute(1, 0)
            r_coord_ = r_coord[:, n:n+1].type(torch.float)

            c_ = torch.matmul(U_t, r_coord_) / self.cfg.max_dist
            r_coord_ap_ = torch.matmul(U, c_) * self.cfg.max_dist

            result = {}
            result['r'] = to_np(r_coord_)[:, 0]
            result['c'] = to_np(c_)[:, 0]
            result['center'] = r_coord_cen[n]
            result['id_xyxy'] = out[i]['id_xyxy']

            self.cen = np.repeat(r_coord_cen[n].reshape(1, -1), self.cfg.node_num, 0).astype(np.float32)

            idx = np.linspace(0, 359, self.cfg.node_num).astype(np.int32)
            theta_list = np.flip(idx, axis=0).astype(np.float32)
            x, y = cv2.polarToCart(to_np(r_coord_ap_.T)[0], theta_list, angleInDegrees=True)  # 360    360
            xy_ap = np.concatenate((x, y), axis=1)
            x, y = cv2.polarToCart(to_np(r_coord_.T)[0], theta_list, angleInDegrees=True)  # 360    360
            xy = np.concatenate((x, y), axis=1)

            polygon_pts_ap = self.cen + xy_ap
            polygon_pts = self.cen + xy

            self.img = self.label_all[i]['cropped_img']
            self.c, self.h, self.w = self.img[0].shape
            self.tmp = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            self.visualize.update_image(self.img[0])
            self.visualize.update_image_name(self.img_name)

            self.visualize.show['ap_mask'] = np.copy(self.visualize.show['img'])
            self.visualize.show['gt_mask'] = np.copy(self.visualize.show['img'])

            img = Image.new("L", (self.w, self.h))
            ImageDraw.Draw(img).polygon(polygon_pts_ap.astype(np.float32), fill=1, outline=True)
            mask_ap = np.array(img)
            self.visualize.draw_mask_cv_2(data=mask_ap, name='ap_mask', ref_name='ap_mask', color=(0, 0, 255))
            self.visualize.draw_polyline_cv(data=polygon_pts_ap, name='ap_mask', ref_name='ap_mask', color=(0, 0, 255))

            img = Image.new("L", (self.w, self.h))
            ImageDraw.Draw(img).polygon(polygon_pts.astype(np.float32), fill=1, outline=True)
            mask = np.array(img)
            # mask = to_np(self.label_all[i]['seg_mask'][0])
            self.visualize.draw_mask_cv_2(data=mask, name='gt_mask', ref_name='gt_mask', color=(0, 255, 0))
            self.visualize.draw_polyline_cv(data=polygon_pts, name='gt_mask', ref_name='gt_mask', color=(0, 255, 0))

            f_measure = batched_f_measure(mask[np.newaxis],
                                          mask_ap[np.newaxis],
                                          average_over_objects=True,
                                          nb_objects=None,
                                          bound_th=0.008)

            self.F[self.cfg.dim] = torch.cat((self.F[self.cfg.dim], torch.tensor(f_measure).type(torch.float32).cuda()))
            results.append(result)
            n += 1

            if self.cfg.display == True :
                self.visualize.display_saveimg_v2(dir_name=dir_, file_name=str(self.cfg.dim) + '_' + str(i) + '.jpg',
                                                  list=['img', 'ap_mask', 'gt_mask'])

        return results

    def make_dict(self):
        self.F = {}
        self.F[self.cfg.dim] = torch.FloatTensor([]).cuda()

    def run(self):
        print('start')
        self.make_dict()

        self.mat = load_pickle(self.cfg.output_dir + 'matrix')
        self.U = load_pickle(self.cfg.output_dir + 'U')
        self.S = load_pickle(self.cfg.output_dir + 'S')

        for i, batch in enumerate(tqdm(self.dataloader)):
            self.label_all = batch['output']
            self.img_name = batch['img_name'][0]

            self.data = load_pickle(self.cfg.output_dir + 'pickle/' + self.img_name)[0]
            out_f = list()
            out_f.append(self.approximate_contour())

            # save data
            if self.cfg.save_pickle == True:
                self.datalist.append(self.img_name)
                save_pickle(dir_name=self.cfg.output_dir + 'pickle_c/',
                            file_name=self.img_name,
                            data=out_f)
                save_pickle(dir_name=self.cfg.output_dir + 'pickle_c/',
                            file_name='datalist',
                            data=self.datalist)
            print('image %d ===> %s clear' % (i, self.img_name))

        if self.cfg.save_pickle == True:
            save_pickle(dir_name=self.cfg.output_dir,
                        file_name='result',
                        data=self.F)