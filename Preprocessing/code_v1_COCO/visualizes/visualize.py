import cv2
from libs.utils import *

class Visualize(object):

    def __init__(self, cfg):

        self.cfg = cfg

        self.mean = np.array([cfg.mean], dtype=np.float32)
        self.std = np.array([cfg.std], dtype=np.float32)

        self.line = np.zeros((cfg.height, 3, 3), dtype=np.uint8)
        self.show = {}

    def update_image(self, img, name='img'):
        img = to_np(img.permute(1, 2, 0))
        img = np.uint8((img * self.std + self.mean) * 255)[:, :, [2, 1, 0]]
        self.show[name] = img

    def update_label(self, label, name='label'):
        label = to_np(label)
        label = np.repeat(np.expand_dims(np.uint8(label != 0) * 255, axis=2), 3, 2)
        self.show[name] = label

    def update_image_name(self, img_name):
        self.show['img_name'] = img_name

    def draw_lines_cv(self, data, name, ref_name='img', color=(255, 0, 0), s=2):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        for i in range(data.shape[0]):
            pts = data[i]
            pt_1 = (pts[0], pts[1])
            pt_2 = (pts[2], pts[3])
            img = cv2.line(img, pt_1, pt_2, color, s)

        self.show[name] = img

    def draw_polyline_cv(self, data, name, ref_name='img', color=(255, 0, 0), s=2):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        pts = np.int32(data).reshape((-1, 1, 2))
        img = cv2.polylines(img, [pts], False, color, s)
        self.show[name] = img

    def draw_points_cv(self, data, name, ref_name='img', color=(0, 255, 0), s=2):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        for i in range(data.shape[0]):
            pts = (int(data[i, 0]), int(data[i, 1]))
            img = cv2.circle(img, pts, s, color, -1)

        self.show[name] = img

    def draw_mask_cv(self, data, name, ref_name='img', color=(255, 0, 0)):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        mask = (np.repeat(data[..., np.newaxis], 3, axis=2) * np.array(color).reshape(1, 1, -1)).astype(np.uint8)
        img += mask
        self.show[name] = img

    def draw_mask_cv_2(self, data, name, ref_name='img', color=(0, 0, 255)):
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
        disp = np.zeros((self.show[list[0]].shape[0], 3, 3), dtype=np.uint8)
        line = disp
        for i in range(len(list)):
            if list[i] not in self.show.keys():
                continue
            disp = np.concatenate((disp, self.show[list[i]], line), axis=1)

        mkdir(dir_name)
        cv2.imwrite(dir_name + file_name + '.png', disp)

    def display_saveimg_v2(self, dir_name, file_name, list):
        disp = np.array([], dtype=np.uint8)
        for name in list:
            if disp.shape[0] == 0:
                disp = self.show[name]
                continue
            disp = np.concatenate((disp, self.show[name]), axis=1)

        mkdir(dir_name)
        cv2.imwrite(os.path.join(dir_name, file_name), disp)


