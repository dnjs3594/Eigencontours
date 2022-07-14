from PIL import Image, ImageDraw
from label_utils.label_centerdeg_custom import *
from label_utils.decoding import *
from libs.utils import *
from tqdm import tqdm

class Generate_Contour(object):

    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.dataloader = dict_DB['dataloader']
        self.visualize = dict_DB['visualize']
        self.size = np.float32([cfg.height, cfg.width])

        self.datalist = []
        self.datalist_error = []

        self.r_coord = np.linspace(0, 360, self.cfg.node_num, endpoint=False)
        self.r_coord_x = np.cos(self.r_coord*2*math.pi/360)[:, np.newaxis]
        self.r_coord_y = np.sin(self.r_coord*2*math.pi/360)[:, np.newaxis]
        self.r_coord_xy = np.concatenate((self.r_coord_x, self.r_coord_y), axis=1)

    def generate_shape(self):
        results = []
        self.error = False

        for i, label in enumerate(self.label_all):
            self.error = False
            self.img = label['cropped_img']
            self.c, self.h, self.w = self.img[0].shape
            self.tmp = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            self.visualize.update_image(self.img[0])
            self.visualize.update_image_name(self.img_name)

            self.visualize.show['polygon_mask'] = np.copy(self.tmp)
            self.visualize.show['ap_mask'] = np.copy(self.tmp)

            result = {}

            # self.label_pts = to_np(label[0])
            self.label_cen = np.round(to_np(label['bbox_center'][0]))
            self.label_bbox = to_np(label['bbox'][0])
            self.label_id = int(label['id'])

            # result['polygon_pts'] = self.label_pts
            # result['center'] = self.label_cen[0]
            result['id_xyxy'] = np.zeros(5, dtype=np.int32)
            result['id_xyxy'][: -1] = self.label_bbox.reshape(-1)
            result['id_xyxy'][-1] = self.label_id

            out = dict()
            out['categ_id'] = self.label_id
            out.update(runOneImage(label['bbox'], label['seg_mask'][0], self.cfg.process_mode))
            self.error = out['check']

            if self.error == False:

                out.update(decoding_theta(out, self.cfg.node_num))
                result['r'] = out['r']
                result['pts'] = out['contour_pts']
                result['center'] = list(out['center'])
                polygon_pts_ap = np.array(result['pts'], dtype=np.float32)
                mask = to_np(label['seg_mask'][0])

                cen = np.repeat(np.array(result['center'])[:, np.newaxis], self.cfg.node_num, 1).T
                xy = np.flip(np.array(out['r']))[:, np.newaxis] * self.r_coord_xy
                polygon_pts_ap = cen + xy

                img = Image.new("L", (self.w, self.h))
                ImageDraw.Draw(img).polygon(np.round(polygon_pts_ap).astype(np.float32), fill=1, outline=True)
                mask_ap = np.array(img)

                mask_overlap = mask + mask_ap
                non_lap = (mask_overlap == 1).astype(np.uint8)
                over_lap = (mask_overlap == 2).astype(np.uint8)
                iou = (over_lap.sum() / (over_lap.sum() + non_lap.sum()))

                if iou >= self.cfg.thresd_iou:
                    self.error = False
                else:
                    self.error = True
                    result['r'] = []
                    result['pts'] = []

                    if self.cfg.display == True:
                        self.visualize.draw_mask_cv(data=mask, name='polygon_mask', ref_name='polygon_mask', color=(0, 0, 255))
                        self.visualize.draw_mask_cv(data=mask_ap, name='ap_mask', ref_name='ap_mask', color=(0, 255, 0))
                        if iou >= self.cfg.thresd_iou:
                            dir_name = self.cfg.output_dir + 'display/'
                        else:
                            dir_name = self.cfg.output_dir + 'display_error/'
                        file_name = self.img_name + '_{}'.format(str(i))
                        self.visualize.display_saveimg(dir_name=dir_name,
                                                       file_name=file_name,
                                                       list=['img', 'polygon_mask', 'ap_mask'])
            else:
                result['r'] = []
                result['pts'] = []

            self.error_all.append(self.error)
            results.append(result)

            if self.cfg.display == True and self.error == False:

                # polygon_pts = self.label_pts

                # img = Image.new("L", (self.w, self.h))
                # ImageDraw.Draw(img).polygon(np.round(polygon_pts).astype(np.float32), fill=1, outline=True)

                # self.visualize.draw_mask_cv(data=non_lap, name='IOU_mask', ref_name='IOU_mask', color=(255, 255, 255))
                # self.visualize.draw_mask_cv(data=over_lap, name='IOU_mask', ref_name='IOU_mask', color=(255, 0, 0))

                self.visualize.draw_mask_cv(data=mask, name='polygon_mask', ref_name='polygon_mask', color=(0, 0, 255))
                self.visualize.draw_mask_cv(data=mask_ap, name='ap_mask', ref_name='ap_mask', color=(0, 255, 0))
                if iou >= self.cfg.thresd_iou:
                    dir_name = self.cfg.output_dir + 'display/'
                else:
                    dir_name = self.cfg.output_dir + 'display_error/'
                file_name = self.img_name + '_{}'.format(str(i))
                self.visualize.display_saveimg(dir_name=dir_name,
                                               file_name=file_name,
                                               list=['img', 'polygon_mask', 'ap_mask'])

        return results

    def run(self):
        print('start')

        for i, batch in enumerate(tqdm(self.dataloader)):
            self.error_all =[]
            self.label_all = batch['output']
            self.img_name = batch['img_name'][0]

            out_f = list()
            out_f.append(self.generate_shape())

            error_all = (~np.array(self.error_all)).sum()
            if error_all != 0 and self.cfg.save_pickle == True:
                self.datalist.append(self.img_name)
                save_pickle(dir_name=self.cfg.output_dir + 'pickle/',
                            file_name=self.img_name,
                            data=out_f)
                save_pickle(dir_name=self.cfg.output_dir + 'pickle/',
                            file_name='datalist',
                            data=self.datalist)

            if error_all == 0 and self.cfg.save_pickle == True:
                self.datalist_error.append(self.img_name)
                save_pickle(dir_name=self.cfg.output_dir + 'pickle/',
                            file_name='datalist_error',
                            data=self.datalist_error)

            print('image %d ===> %s clear, error_all %d' % (i, self.img_name, error_all))