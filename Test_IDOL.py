import math
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from Dataset import *
import Config as config
transform_test = None

def test_model(model, dataset):

    model.eval()

    msw_error = 0
    msw_RMSE_sum = 0
    rmw_error = 0
    rmw_RMSE_sum = 0
    r34_error = 0
    r34_RMSE_sum = 0
    mslp_error = 0
    mslp_RMSE_sum = 0

    msw_output_list = []
    msw_label_list = []
    rmw_output_list = []
    rmw_label_list = []
    r34_output_list = []
    r34_label_list = []
    mslp_output_list = []
    mslp_label_list = []

    v_errors = []
    p_errors = []
    r1_errors = []
    r2_errors = []

    with torch.no_grad():
        for batch, data in enumerate(tqdm(dataset)):
            '''5, 156, 156'''
            k8_btemp = data["k8_btemp"]
            k8_btemp = k8_btemp.to(config.device)

            pxh_k8_btemp = data["pxh_k8_btemp"]
            pxh_k8_btemp = pxh_k8_btemp.to(config.device)
            '''cat pxh and now'''
            btemp = torch.cat([pxh_k8_btemp, k8_btemp], dim=1)

            pre_level = data["pre_level"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            norm_t = data['norm_t'].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            plt = torch.cat([pre_level, norm_t], dim=-1)

            pre_tcf = data["pre_tcf"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            pre_isr = data["pre_isr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            pre_pwr = data["pre_pwr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            pre_prr = data["pre_prr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            pre_pr = torch.cat([pre_tcf, pre_isr, pre_pwr, pre_prr], dim=-1)

            msw_label = data["msw"]
            msw_label = msw_label.to(config.device)
            rmw_label = data["rmw"]
            rmw_label = rmw_label.to(config.device)
            mslp_label = data["mslp"]
            mslp_label = mslp_label.to(config.device)
            r34_label = data["r34"]
            r34_label = r34_label.to(config.device)
            TCA_Lable = [i.unsqueeze(1) for i in [msw_label, mslp_label, rmw_label, r34_label]]
            TCA_Lable = torch.cat(TCA_Lable, dim=1)

            msw, mslp, rmw, r34, loss_simr = model(btemp, plt, pre_pr, TCA_Lable)

            print("msw_label={}".format(msw_label))
            print("msw={}".format(msw))
            print("mslp_label={}".format(mslp_label))
            print("mslp={}".format(mslp))
            print("rmw_label={}".format(rmw_label))
            print("rmw={}".format(rmw))
            print("r34_label={}".format(r34_label))
            print("r34={}".format(r34))

            msw_label_re = msw_label.cpu().detach().numpy() * (170 - 35) + 35
            msw_re = msw.cpu().detach().numpy() * (170 - 35) + 35
            msw_error = msw_error + np.sum(np.abs(msw_re - msw_label_re))
            msw_RMSE_sum = msw_RMSE_sum + np.sum((msw_re - msw_label_re) * (msw_re - msw_label_re))

            mslp_label_re = mslp_label.cpu().detach().numpy() * (1010 - 882) + 882
            mslp_re = mslp.cpu().detach().numpy() * (1010 - 882) + 882
            mslp_error = mslp_error + np.sum(np.abs(mslp_re - mslp_label_re))
            mslp_RMSE_sum = mslp_RMSE_sum + np.sum((mslp_re - mslp_label_re) * (mslp_re - mslp_label_re))

            rmw_label_re = rmw_label.cpu().detach().numpy() * (130 - 5) + 5
            rmw_re = rmw.cpu().detach().numpy() * (130 - 5) + 5
            rmw_error = rmw_error + np.sum(np.abs(rmw_re - rmw_label_re))
            rmw_RMSE_sum = rmw_RMSE_sum + np.sum((rmw_re - rmw_label_re) * (rmw_re - rmw_label_re))

            r34_label_re = r34_label.cpu().detach().numpy() * (406.25 - 10) + 10
            r34_re = r34.cpu().detach().numpy() * (406.25 - 10) + 10
            r34_error = r34_error + np.sum(np.abs(r34_re - r34_label_re))
            r34_RMSE_sum = r34_RMSE_sum + np.sum((r34_re - r34_label_re) * (r34_re - r34_label_re))

            for i in range(0, len(msw_re)):
                msw_output_list.append(msw_re[i])
                msw_label_list.append(msw_label_re[i])
                mslp_output_list.append(mslp_re[i])
                mslp_label_list.append(mslp_label_re[i])
                rmw_output_list.append(rmw_re[i])
                rmw_label_list.append(rmw_label_re[i])
                r34_output_list.append(r34_re[i])
                r34_label_list.append(r34_label_re[i])
                v_errors.append(np.abs(msw_re[i] - msw_label_re[i]))
                p_errors.append(np.abs(mslp_re[i] - mslp_label_re[i]))
                r1_errors.append(np.abs(rmw_re[i] - rmw_label_re[i]))
                r2_errors.append(np.abs(r34_re[i] - r34_label_re[i]))
    v_err_mean, v_err_var = np.mean(v_errors), np.std(v_errors)
    p_err_mean, p_err_var = np.mean(p_errors), np.std(p_errors)
    r1_err_mean, r1_err_var = np.mean(r1_errors), np.std(r1_errors)
    r2_err_mean, r2_err_var = np.mean(r2_errors), np.std(r2_errors)
    print("estimation v mean = {}, std = {}".format(v_err_mean, v_err_var))
    print("estimation p mean = {}, std = {}".format(p_err_mean, p_err_var))
    print("estimation r1 mean = {}, std = {}".format(r1_err_mean, r1_err_var))
    print("estimation r2 mean = {}, std = {}".format(r2_err_mean, r2_err_var))
    return msw_error, msw_RMSE_sum, msw_label_list, msw_output_list, \
           mslp_error, mslp_RMSE_sum, mslp_label_list, mslp_output_list, \
           rmw_error, rmw_RMSE_sum, rmw_label_list, rmw_output_list, \
           r34_error, r34_RMSE_sum, r34_label_list, r34_output_list

if __name__ == '__main__':
    model = torch.load(config.predict_model).to(config.device)

    test_transform = None

    test_dataset = Findpxh_Dataset_k_pr(config.predict_k8_path, 3, None, config.data_format)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
    )

    msw_error, msw_RMSE_sum, msw_label_list, msw_output_list, \
    mslp_error, mslp_RMSE_sum, mslp_label_list, mslp_output_list, \
    rmw_error, rmw_RMSE_sum, rmw_label_list, rmw_output_list, \
    r34_error, r34_RMSE_sum, r34_label_list, r34_output_list = test_model(model, test_dataloader)

    test_num = len(msw_label_list)
    print("test sample={}".format(test_num))

    test_msw_error = msw_error / test_num
    msw_RMSE = math.sqrt(msw_RMSE_sum / test_num)
    print(f'Mean MSW MAE: {test_msw_error:.2f}')
    print(f'Mean MSW RMSE: {msw_RMSE:.2f}')

    test_rmw_error = rmw_error / test_num
    rmw_RMSE = math.sqrt(rmw_RMSE_sum / test_num)
    print(f'Mean RMW MAE: {test_rmw_error:.2f}')
    print(f'Mean RMW RMSE: {rmw_RMSE:.2f}')

    test_mslp_error = mslp_error / test_num
    mslp_RMSE = math.sqrt(mslp_RMSE_sum / test_num)
    print(f'Mean mslp MAE: {test_mslp_error:.2f}')
    print(f'Mean mslp RMSE: {mslp_RMSE:.2f}')

    test_r34_error = r34_error / test_num
    r34_RMSE = math.sqrt(r34_RMSE_sum / test_num)
    print(f'Mean R34 MAE: {test_r34_error:.2f}')
    print(f'Mean R34 RMSE: {r34_RMSE:.2f}')

