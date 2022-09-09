
import torch
import numpy as np
import os
import pickle
import math
from mmdet.visualization.visualize import Visualize_cv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def mkdir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)

def save_pickle(dir_name, file_name, data):

    '''
    :param file_path: ...
    :param data:
    :return:
    '''
    mkdir(dir_name)
    with open(dir_name + file_name + '.pickle', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(file_path):
    with open(file_path + '.pickle', 'rb') as f:
        data = pickle.load(f)

    return data


class SVD(object):
    def __init__(self):
        self.height = 768
        self.width = 1280
        self.dist = math.sqrt(self.height * self.height / 4 + self.width * self.width / 4) / 10

        self.center = np.array([(640 - 1) / 2, (640 - 1) / 2])
        self.r_coord = np.linspace(0, 360, 360, endpoint=False)
        self.r_coord_x = torch.tensor(np.cos(self.r_coord*2*math.pi/360)).view(-1, 1)
        self.r_coord_y = torch.tensor((-1) *np.sin(self.r_coord*2*math.pi/360)).view(-1, 1)
        self.r_coord_xy = torch.cat((self.r_coord_x, self.r_coord_y), dim=1).type(torch.float32).cuda()
        self.cen = torch.tensor(np.repeat(self.center.reshape(1, -1), 360, 0)).type(
            torch.float32).cuda()


    def update_matrix(self, data):
        self.mat = torch.cat((self.mat, data.transpose(1, 0)), dim=1)

    def load_contour_component(self):
        f_list = os.listdir('data_pickle_f_sbd_360_re_')
        nf_list = os.listdir('data_pickle_nf_sbd_360_re_')

        f_list = [i.rstrip('.pickle') for i in f_list]
        nf_list = [i.rstrip('.pickle') for i in nf_list]

        # sampled
        for i, name in enumerate(f_list):
            data_p = load_pickle('data_pickle_f_sbd_360_re_/' + name)
            if len(data_p) != 0:
                self.update_matrix(data_p.cuda())
            print('%d done!' % i)

        for i, name in enumerate(nf_list):
            data_p = load_pickle('data_pickle_nf_sbd_360_re_/' + name)
            if len(data_p) != 0:
                self.update_matrix(data_p.cuda())
            print('%d done!' % i)

        save_pickle(dir_name='data_pickle/',
                    file_name='matrix_re_',
                    data=self.mat)

    def do_SVD(self):
        self.mat = load_pickle('data_pickle/matrix_re')

        n, l = self.mat.shape
        idx = torch.linspace(0, l-1, 40000).type(torch.int64).cuda()
        U, S, V = np.linalg.svd(self.mat.cpu().numpy() / (self.dist), full_matrices=True)
        self.U = torch.from_numpy(U).cuda()
        self.S = torch.from_numpy(S).cuda()

        save_pickle(dir_name='data_pickle/',
                    file_name='U_re',
                    data=self.U)
        save_pickle(dir_name='data_pickle/',
                    file_name='S_re',
                    data=self.S)

    def make_dict(self):
        self.mat = torch.FloatTensor([]).cuda()
        self.U = torch.FloatTensor([]).cuda()
        self.S = torch.FloatTensor([]).cuda()
        self.V = torch.FloatTensor([]).cuda()

    def visualization_U(self):
        self.U = load_pickle("data_pickle/U")
        for k in range(self.U.shape[1]):

            if k == 6:
                break
            temp = np.full((640, 640, 3), fill_value=255, dtype=np.uint8)
            self.visualize.show['candidates'] = np.copy(temp)

            U = self.U[:, k:k + 1] * 3000
            dc = torch.full((360, 1), fill_value=180).type(torch.float32).cuda()
            xy = self.r_coord_xy * U
            xy_dc = self.r_coord_xy * dc
            polygon_pts = self.cen + xy
            polygon_pts_dc = self.cen + xy_dc

            allow_pts = torch.cat((self.cen, polygon_pts), dim=1)
            # self.visualize.draw_arrowedlines_cv(data=to_np(allow_pts).astype(np.int64), name='candidates', interval=1, ref_name='candidates',
            #                                 color=(255, 255, 255), s=1)
            self.visualize.draw_polyline_cv(data=polygon_pts.cpu().numpy(), name='candidates', ref_name='candidates',
                                            color=(0, 0, 255), s=15)
            # self.visualize.draw_polyline_cv(data=polygon_pts.numpy(), name='candidates', ref_name='candidates',
            #                                 color=(51, 153, 102), s=15)
            # self.visualize.draw_points_cv(data=to_np(self.cen[0:1, :]), name='candidates', ref_name='candidates',
            #                                 color=(255, 255, 255), s=5)
            dir_name = 'display_U_red/'
            file_name = 'U_' + str(k + 1) + '.jpg'
            # file_name = 'U_' + str(k) + '.png'
            self.visualize.display_saveimg(dir_name=dir_name,
                                           file_name=file_name,
                                           list=['candidates'])


    def run(self):
        print('start')
        self.make_dict()
        self.visualize = Visualize_cv()
        # self.load_contour_component()
        # self.do_SVD()
        self.visualization_U()

if __name__ == "__main__":
    svd = SVD()
    svd.run()