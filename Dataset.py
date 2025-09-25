import numpy as np
import torch
import os
from tqdm import tqdm
import Config as config
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt

def find_previous_hours(time_str, x):
    # Parse time string
    time_format = "%Y%m%d%H"
    time = datetime.strptime(time_str, time_format)

    previous_time = time - timedelta(hours=x)

    # transform to string
    previous_time_str = previous_time.strftime(time_format)

    return previous_time_str

def get_previous_npy_index(npy_index, x):
    # parse npy_index
    parts = npy_index.split('_')
    year_typhoon_name = parts[0] + '_' + parts[1]
    time_str = parts[2]  # 例如 '2019010506'

    time_format = '%Y%m%d%H'
    time_obj = datetime.strptime(time_str, time_format)

    previous_time_obj = time_obj - timedelta(hours=x)

    previous_time_str = previous_time_obj.strftime(time_format)

    previous_npy_index = year_typhoon_name + "_" + previous_time_str

    return previous_npy_index

def default_loader(path):
    raw_data = np.load(path, allow_pickle=True)
    tensor_data = torch.from_numpy(raw_data)
    tensor_data = tensor_data.type(torch.FloatTensor)
    return tensor_data

def load_dict_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def get_fnorm_now_chw(x, statistic_dic):
    x_norm = x.clone()
    for c in range(4):
        maxv = statistic_dic['now_all'][c][0]
        minv = statistic_dic['now_all'][c][1]
        x_norm[c] = (x[c] - minv) / (maxv - minv)

    return x_norm

def get_fnorm_p3h_chw(x, statistic_dic):
    x_norm = x.clone()
    for c in range(4):
        maxv = statistic_dic['p3h_all'][c][0]
        minv = statistic_dic['p3h_all'][c][1]
        x_norm[c] = (x[c] - minv) / (maxv - minv)
    return x_norm

class Findpxh_Dataset_k_pr():
    def __init__(self, k8_path, pxh, data_transforms=None, data_format='npy'):

        self.data_transforms = data_transforms
        self.data_format = data_format
        self.k8_paths = k8_path
        self.pxh = pxh
        self.k8_btemps = []
        self.pxh_k8_btemps = []

        self.msws = []
        self.rmws = []
        self.r34s = []
        self.mslps = []

        self.lats = []
        self.lons = []
        self.ts = []
        self.levels = []
        self.plevels = []

        self.pre_tcfs = []
        self.pre_isrs = []
        self.pre_pwrs = []
        self.pre_prrs = []
        self.pre_levels = []
        self.norm_ts = []
        self.labels_dic = load_dict_from_pickle(config.labels_path)

        k8_files = os.listdir(k8_path)
        k8_files_set = set(k8_files)

        self.k8_sta_dic = load_dict_from_pickle(config.k8_sta_path)
        pre12h_labels_data = load_dict_from_pickle(config.p12hpr_pth)

        for i, filename in enumerate(k8_files):
            fname_split = filename.split("_")
            tcname = fname_split[1]
            year = fname_split[0]
            '''e.g. 2023_BOLAVEN_2023100809'''
            if len(fname_split) == 3:
                isotime = fname_split[2][:-4]
            else:
                isotime = fname_split[2]
            labels_dic_key = year + "_" + tcname + "_" + isotime

            '''e.g. BOLAVEN_2023100806'''
            p3h_labels_dic_key = get_previous_npy_index(labels_dic_key, 3)
            if len(fname_split) == 3:
                pxh_k8_fname = p3h_labels_dic_key + ".npy"
            else:
                pxh_k8_fname = p3h_labels_dic_key + "_" + fname_split[3]
            if filename in self.k8_sta_dic['now_inval_record'] or pxh_k8_fname in self.k8_sta_dic['p3h_inval_record']:
                continue
            if pxh_k8_fname not in k8_files_set:
                continue
            if p3h_labels_dic_key not in self.labels_dic.keys():
                continue
            if labels_dic_key not in self.labels_dic.keys():
                continue
            dic_index = str(isotime[:4]) + "_" + tcname + "_" + str(isotime)  # 例如2019_台风名字_2019010106
            if dic_index not in pre12h_labels_data.keys():
                continue

            lat, lon, t, level, mslp, msw, rmw, r34 = self.labels_dic[labels_dic_key][1:]
            pre_tcf, pre_isr, pre_pwr, pre_prr, pre_level, pre_mslp, pre_msw, pre_rmw, pre_r34 = pre12h_labels_data[
                dic_index]
            if 0.0 in [mslp, msw, rmw, r34]:
                continue
            self.pre_tcfs.append(pre_tcf)
            self.pre_isrs.append(pre_isr)
            self.pre_pwrs.append(pre_pwr)
            self.pre_prrs.append(pre_prr)
            self.plevels.append(pre_level)

            self.pxh_k8_btemps.append(k8_path + pxh_k8_fname)
            self.k8_btemps.append(k8_path + filename)
            self.msws.append(msw)
            self.rmws.append(rmw)
            self.r34s.append(r34)
            self.mslps.append(mslp)

            self.lats.append(lat)
            self.lons.append(lon)
            self.ts.append(t)
            self.levels.append(level)
            self.norm_ts.append(t)

        print("msws max = {}, min = {}".format(max(self.msws), min(self.msws)))
        print("rmws max = {}, min = {}".format(max(self.rmws), min(self.rmws)))
        print("r34s max = {}, min = {}".format(max(self.r34s), min(self.r34s)))
        print("mslps max = {}, min = {}".format(max(self.mslps), min(self.mslps)))
        print("lat max = {}, min = {}".format(max(self.lats), min(self.lats)))
        print("lon max = {}, min = {}".format(max(self.lons), min(self.lons)))
        print("t max = {}, min = {}".format(max(self.ts), min(self.ts)))
        print("pre_tcf max = {}, min = {}".format(max(self.pre_tcfs), min(self.pre_tcfs)))
        print("pre_isr max = {}, min = {}".format(max(self.pre_isrs), min(self.pre_isrs)))
        print("pre_pwr max = {}, min = {}".format(max(self.pre_pwrs), min(self.pre_pwrs)))
        print("pre_prr max = {}, min = {}".format(max(self.pre_prrs), min(self.pre_prrs)))

        for i in range(len(self.msws)):
            self.msws[i] = (self.msws[i] - 35) / (170 - 35)
            self.rmws[i] = (self.rmws[i] - 5) / (130 - 5)
            self.r34s[i] = (self.r34s[i] - 10) / (406.25 - 10)
            self.mslps[i] = (self.mslps[i] - 882) / (1010 - 882)
            self.lats[i] = (self.lats[i] - (-32.038)) / (42.491 - (-32.038))
            self.lons[i] = (self.lons[i] - 196.1) / (196.1 - 83.892)
            self.norm_ts[i] = (self.ts[i] - 12) / (459 - 12)
            self.pre_tcfs[i] = (self.pre_tcfs[i] - (-2)) / (0.98 - (-2))
            self.pre_isrs[i] = (self.pre_isrs[i] - 0.22) / (34 - 0.22)
            self.pre_pwrs[i] = (self.pre_pwrs[i] - (5.18)) / (28.77 - (5.18))
            self.pre_prrs[i] = (self.pre_prrs[i] - (2.76)) / (99.9 - (2.76))

    def __len__(self):
        return len(self.k8_btemps)

    def __getitem__(self, index):
        # 4, 156, 156
        btemp_file_path = self.k8_btemps[index]
        k8_btemp = default_loader(btemp_file_path)
        k8_btemp = get_fnorm_now_chw(k8_btemp, self.k8_sta_dic)

        # 4, 156, 156
        pxh_btemp_file_path = self.pxh_k8_btemps[index]
        pxh_k8_btemp = default_loader(pxh_btemp_file_path)
        pxh_k8_btemp = get_fnorm_p3h_chw(pxh_k8_btemp, self.k8_sta_dic)

        msw = self.msws[index]
        rmw = self.rmws[index]
        r34 = self.r34s[index]
        mslp = self.mslps[index]

        lat = self.lats[index]
        lon = self.lons[index]
        level = self.levels[index]
        t = self.ts[index]
        norm_t = self.norm_ts[index]
        pre_tcf = round(self.pre_tcfs[index], 3)
        pre_isr = round(self.pre_isrs[index], 3)
        pre_pwr = round(self.pre_pwrs[index], 3)
        pre_prr = round(self.pre_prrs[index], 3)
        pre_level = self.plevels[index]

        sample = {'lat': lat, 'lon': lon, 'occur_t': t, 'cat': level, 'norm_t': norm_t,
                  'pre_level': pre_level, 'r34': r34, 'rmw': rmw, 'msw': msw, 'mslp': mslp,
                  'pre_tcf': pre_tcf, 'pre_isr': pre_isr, 'pre_pwr': pre_pwr, 'pre_prr': pre_prr,
                  'k8_btemp': k8_btemp,
                  'pxh_k8_btemp': pxh_k8_btemp
                  }
        return sample

if __name__ == '__main__':
    '''find pxh npy in single t npy'''
    print("trainset label max/min v:")
    # train_dataset = Findpxh_Dataset_k('/opt/data/private/norm_data_npy/Full2015_2023_Dataets/k89_4ch1/train/', 3, None, config.data_format)
    train_dataset = Findpxh_Dataset_k_pr(config.train_k8_path, 3, None, config.data_format)
    print("validset label max/min v:")
    valid_dataset = Findpxh_Dataset_k_pr(config.valid_k8_path, 3, None, config.data_format)
    print("testset label max/min v:")
    test_dataset = Findpxh_Dataset_k_pr(config.predict_k8_path, 3, None, config.data_format)
    # for batch, data in enumerate(tqdm(test_dataset)):
    #     k8_btemp = data["k8_btemp"]
        # print("k8_btemp shape", k8_btemp.size())
        # pxh_k8_btemp = data["pxh_k8_btemp"]



