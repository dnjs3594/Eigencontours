from libs.utils import *
import numpy as np
import torch

class SVD(object):

    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.dataloader = dict_DB['dataloader']
        self.visualize = dict_DB['visualize']
        self.height = cfg.height
        self.width = cfg.width
        self.size = np.float32([cfg.height, cfg.width])

        self.center = np.array([(self.width - 1) / 2, (self.height - 1) / 2])
        self.datalist = []

    def update_matrix(self, data):
        self.mat = torch.cat((self.mat, data.view(-1, 1)), dim=1)

    def load_contour_component(self):
        datalist = load_pickle(self.cfg.output_dir + 'pickle/datalist')

        # sampled
        for i in range(len(datalist)):

            img_name = datalist[i]
            data_p = load_pickle(self.cfg.output_dir + 'pickle/' + img_name)[0]

            for k in range(len(data_p)):
                if len(data_p[k]['r']) != 0:
                    self.update_matrix(torch.tensor(data_p[k]['r']).type(torch.float32).cuda())

            print('%d done!' % i)

        if self.cfg.save_pickle == True:
            save_pickle(dir_name=self.cfg.output_dir,
                        file_name='matrix',
                        data=self.mat)

    def do_SVD(self):
        self.mat = load_pickle(self.cfg.output_dir + 'matrix')

        n, l = self.mat.shape
        # idx = torch.linspace(0, l-1, 40000).type(torch.int64).cuda()
        U, S, V = np.linalg.svd(self.mat[:, idx].cpu().numpy() / (self.cfg.max_dist), full_matrices=False)
        self.U = torch.from_numpy(U).cuda()
        self.S = torch.from_numpy(S).cuda()
        # self.V = V.cuda()

        if self.cfg.save_pickle == True:
            save_pickle(dir_name=self.cfg.output_dir,
                        file_name='U',
                        data=self.U)
            save_pickle(dir_name=self.cfg.output_dir,
                        file_name='S',
                        data=self.S)

    def make_dict(self):
        self.mat = torch.FloatTensor([]).cuda()
        self.U = torch.FloatTensor([]).cuda()
        self.S = torch.FloatTensor([]).cuda()
        self.V = torch.FloatTensor([]).cuda()

    def run(self):
        print('start')
        self.make_dict()
        self.load_contour_component()
        self.do_SVD()
