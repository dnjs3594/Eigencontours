import torch
import numpy as np
import os
import pickle
import math

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
                    file_name='matrix_re',
                    data=self.mat)

    def do_SVD(self):
        self.mat = load_pickle( 'data_pickle/matrix_re')

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

    def run(self):
        print('start')
        self.make_dict()
        self.load_contour_component()
        self.do_SVD()

if __name__ == "__main__":
    svd = SVD()
    svd.run()