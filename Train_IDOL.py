import math
import os
import torch.optim
from torch import nn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import Set_seed
from Dataset import *
import Config as config
from IDOL import *

def save_model(model, model_output_dir, epoch):
    save_model_file = os.path.join(model_output_dir, "epoch_{}.pth".format(epoch))
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    torch.save(model, save_model_file)

def train_model(model, loss_func, dataset, optimizer, w_params, epoch):

    model.train()

    batch_msw_loss = 0
    batch_rmw_loss = 0
    batch_r34_loss = 0
    batch_mslp_loss = 0

    batch_simr_loss = 0

    batch_loss = 0
    item = 0
    batch_num = 0

    msw_error = 0
    rmw_error = 0
    r34_error = 0
    mslp_error = 0

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

        optimizer.zero_grad()

        msw, mslp, rmw, r34, loss_simr = model(btemp, plt, pre_pr, TCA_Lable)

        print("msw_label={}".format(msw_label))
        print("msw={}".format(msw))
        print("mslp_label={}".format(mslp_label))
        print("mslp={}".format(mslp))
        print("rmw_label={}".format(rmw_label))
        print("rmw={}".format(rmw))
        print("r34_label={}".format(r34_label))
        print("r34={}".format(r34))

        msw_loss = loss_func(msw.float(), msw_label.float())
        mslp_loss = loss_func(mslp.float(), mslp_label.float())
        rmw_loss = loss_func(rmw.float(), rmw_label.float())
        r34_loss = loss_func(r34.float(), r34_label.float())

        total_loss = w_params[0] * msw_loss + w_params[1] * mslp_loss + w_params[2] * rmw_loss + w_params[3] * r34_loss\
                     + 0.5 * loss_sp

        total_loss.backward()
        optimizer.step()

        # monitor gradient
        if batch % 5 == 0:
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = round(param.grad.data.norm(2).item(), 5)
                    print(
                        f'Epoch [{epoch}], Step [{batch + 1}/{len(dataset)}], Gradient norm: {grad_norm}')
                    
        print("Train Epoch = {} loss_simr = {}".format(epoch, loss_simr.data.item()))
        print("Train Epoch = {} msw Loss = {}".format(epoch, msw_loss.data.item()))
        print("Train Epoch = {} mslp Loss = {}".format(epoch, mslp_loss.data.item()))
        print("Train Epoch = {} rmw Loss = {}".format(epoch, rmw_loss.data.item()))
        print("Train Epoch = {} r34 Loss = {}".format(epoch, r34_loss.data.item()))
        print("Train Epoch = {} Loss = {}".format(epoch, total_loss.data.item()))
        batch_loss += total_loss.data.item()

        batch_simr_loss += loss_simr.item()

        batch_msw_loss += msw_loss.data.item()
        batch_mslp_loss += mslp_loss.data.item()
        batch_rmw_loss += rmw_loss.data.item()
        batch_r34_loss += r34_loss.data.item()

        msw_label_re = msw_label.cpu().detach().numpy() * (170 - 35) + 35
        msw_re = msw.cpu().detach().numpy() * (170 - 35) + 35
        msw_error = msw_error + np.sum(np.abs(msw_re - msw_label_re))

        mslp_label_re = mslp_label.cpu().detach().numpy() * (1010 - 882) + 882
        mslp_re = mslp.cpu().detach().numpy() * (1010 - 882) + 882
        mslp_error = mslp_error + np.sum(np.abs(mslp_re - mslp_label_re))

        rmw_label_re = rmw_label.cpu().detach().numpy() * (130 - 5) + 5
        rmw_re = rmw.cpu().detach().numpy() * (130 - 5) + 5
        rmw_error = rmw_error + np.sum(np.abs(rmw_re - rmw_label_re))

        r34_label_re = r34_label.cpu().detach().numpy() * (406.25 - 10) + 10
        r34_re = r34.cpu().detach().numpy() * (406.25 - 10) + 10
        r34_error = r34_error + np.sum(np.abs(r34_re - r34_label_re))

        batch_num += 1
        item += len(msw_re)

    print("Train Epoch = {} mean loss = {} ".format(epoch, batch_loss / batch_num))
    print("Train Epoch = {} mean loss_sh = {} ".format(epoch, batch_simr_loss / batch_num))
    print("Train Epoch = {} mean msw error = {} ".format(epoch, msw_error / item))
    print("Train Epoch = {} mean mslp error = {} ".format(epoch, mslp_error / item))
    print("Train Epoch = {} mean rmw error = {} ".format(epoch, rmw_error / item))
    print("Train Epoch = {} mean r34 error = {} ".format(epoch, r34_error / item))

    return batch_loss / batch_num, msw_error / item, mslp_error / item, rmw_error / item, r34_error / item, \
           batch_simr_loss / batch_num

def valid_model(model, loss_func, dataset, epoch):

    model.eval()

    batch_loss = 0
    batch_simr_loss = 0

    batch_num = 0
    item = 0

    msw_error = 0
    mslp_error = 0
    rmw_error = 0
    r34_error = 0

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

            msw_loss = loss_func(msw.float(), msw_label.float())
            mslp_loss = loss_func(mslp.float(), mslp_label.float())
            rmw_loss = loss_func(rmw.float(), rmw_label.float())
            r34_loss = loss_func(r34.float(), r34_label.float())

            total_loss = w_params[0] * msw_loss + w_params[1] * mslp_loss + \
                         w_params[2] * rmw_loss + w_params[3] * r34_loss\
                         + 0.5 * loss_sp
          
            print("Valid Epoch = {} loss_simr = {}".format(epoch, loss_simr.data.item()))
            print("Valid Epoch = {} msw Loss = {}".format(epoch, msw_loss.data.item()))
            print("Valid Epoch = {} mslp Loss = {}".format(epoch, mslp_loss.data.item()))
            print("Valid Epoch = {} rmw Loss = {}".format(epoch, rmw_loss.data.item()))
            print("Valid Epoch = {} r34 Loss = {}".format(epoch, r34_loss.data.item()))

            batch_loss += total_loss.data.item()
            batch_simr_loss += loss_sp.item()

            msw_label_re = msw_label.cpu().detach().numpy() * (170 - 35) + 35
            msw_re = msw.cpu().detach().numpy() * (170 - 35) + 35
            msw_error = msw_error + np.sum(np.abs(msw_re - msw_label_re))

            mslp_label_re = mslp_label.cpu().detach().numpy() * (1010 - 882) + 882
            mslp_re = mslp.cpu().detach().numpy() * (1010 - 882) + 882
            mslp_error = mslp_error + np.sum(np.abs(mslp_re - mslp_label_re))

            rmw_label_re = rmw_label.cpu().detach().numpy() * (130 - 5) + 5
            rmw_re = rmw.cpu().detach().numpy() * (130 - 5) + 5
            rmw_error = rmw_error + np.sum(np.abs(rmw_re - rmw_label_re))

            r34_label_re = r34_label.cpu().detach().numpy() * (406.25 - 10) + 10
            r34_re = r34.cpu().detach().numpy() * (406.25 - 10) + 10
            r34_error = r34_error + np.sum(np.abs(r34_re - r34_label_re))

            batch_num += 1
            item += len(msw_re)

    print("Valid Epoch = {} mean loss = {} ".format(epoch, batch_loss / batch_num))
    print("Valid Epoch = {} mean loss_sh = {} ".format(epoch, batch_simr_loss / batch_num))
    print("Valid Epoch = {} mean msw error = {} ".format(epoch, msw_error / item))
    print("Valid Epoch = {} mean mslp error = {} ".format(epoch, mslp_error / item))
    print("Valid Epoch = {} mean rmw error = {} ".format(epoch, rmw_error / item))
    print("Valid Epoch = {} mean r34 error = {} ".format(epoch, r34_error / item))

    return batch_loss / batch_num, msw_error / item, mslp_error / item, rmw_error / item, r34_error / item, \
           batch_simr_loss / batch_num

def train(model, Estim_Loss, optimizer, w_params, epochs):
    train_transform = None
    valid_transform = None
    
    train_dataset = Findpxh_Dataset_k_pr(config.train_k8_path, 3, None, config.data_format)
    valid_dataset = Findpxh_Dataset_k_pr(config.valid_k8_path, 3, None, config.data_format)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=True
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=True
    )

    '''
    training model
    '''
    start_epoch = 0
    train_loss_array = np.zeros(epochs + 1)
    valid_loss_array = np.zeros(epochs + 1)

    train_simr_array = np.zeros(epochs + 1)
    valid_simr_array = np.zeros(epochs + 1)

    train_msw_error_array = np.zeros(epochs + 1)
    valid_msw_error_array = np.zeros(epochs + 1)
    train_mslp_error_array = np.zeros(epochs + 1)
    valid_mslp_error_array = np.zeros(epochs + 1)
    train_rmw_error_array = np.zeros(epochs + 1)
    valid_rmw_error_array = np.zeros(epochs + 1)
    train_r34_error_array = np.zeros(epochs + 1)
    valid_r34_error_array = np.zeros(epochs + 1)

    for epoch in range(start_epoch + 1, epochs + 1):
        train_epoch_loss, train_msw_error, train_mslp_error, train_rmw_error, train_r34_error, train_simr_loss = \
            train_model(model, Estim_Loss, train_dataloader, optimizer, w_params, epoch)
        valid_epoch_loss, valid_msw_error, valid_mslp_error, valid_rmw_error, valid_r34_error, valid_simr_loss = \
            valid_model(model, Estim_Loss, valid_dataloader, epoch)

        train_loss_array[epoch] = train_epoch_loss
        valid_loss_array[epoch] = valid_epoch_loss
        train_simr_array[epoch] = train_simr_loss
        valid_simr_array[epoch] = valid_simr_loss

        train_msw_error_array[epoch] = train_msw_error
        valid_msw_error_array[epoch] = valid_msw_error
        train_mslp_error_array[epoch] = train_mslp_error
        valid_mslp_error_array[epoch] = valid_mslp_error
        train_rmw_error_array[epoch] = train_rmw_error
        valid_rmw_error_array[epoch] = valid_rmw_error
        train_r34_error_array[epoch] = train_r34_error
        valid_r34_error_array[epoch] = valid_r34_error

        # saving model
        if epoch % config.save_model_iter == 0:
            save_model(model, config.model_output_dir, epoch)

    # paint loss
    fig_loss, ax_loss = plt.subplots(figsize=(12, 8))
    train_loss, = plt.plot(np.arange(1, epochs + 1), train_loss_array[1:], 'r')
    val_loss, = plt.plot(np.arange(1, epochs + 1), valid_loss_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('Estimation loss')
    plt.title("train/valid loss vs epoch")
    ax_loss.legend(handles=[train_loss, val_loss], labels=['train_epoch_loss', 'val_epoch_loss'],
                         loc='best')
    fig_loss.savefig(config.save_fig_dir + 'loss.png')
    plt.close(fig_loss)

    fig_simr, ax_simr = plt.subplots(figsize=(12, 8))
    train_simr, = plt.plot(np.arange(1, epochs + 1), train_simr_array[1:], 'r')
    val_simr, = plt.plot(np.arange(1, epochs + 1), valid_simr_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('simr loss')
    plt.title("train/valid simr loss vs epoch")
    ax_simr.legend(handles=[train_simr, val_simr], labels=['train_epoch_loss', 'val_epoch_loss'],
                   loc='best')
    fig_simr.savefig(config.save_fig_dir + 'simr_loss.png')
    plt.close(fig_simr)

    fig_msw_error, ax_msw_error = plt.subplots(figsize=(12, 8))
    train_msw_error, = plt.plot(np.arange(1, epochs + 1),
                                                      train_msw_error_array[1:], 'r')
    valid_msw_error, = plt.plot(np.arange(1, epochs + 1),
                                       valid_msw_error_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('msw error')
    plt.title("train/valid error vs epoch")
    ax_msw_error.legend(handles=[train_msw_error,
                              valid_msw_error],
                     labels=['train_error', 'valid_error'],
                     loc='best')
    fig_msw_error.savefig(config.save_fig_dir + 'msw_error.png')
    plt.close(fig_msw_error)

    fig_mslp_error, ax_mslp_error = plt.subplots(figsize=(12, 8))
    train_mslp_error, = plt.plot(np.arange(1, epochs + 1),
                                 train_mslp_error_array[1:], 'r')
    valid_mslp_error, = plt.plot(np.arange(1, epochs + 1),
                                 valid_mslp_error_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('mslp error')
    plt.title("train/valid error vs epoch")
    ax_mslp_error.legend(handles=[train_mslp_error,
                                 valid_mslp_error],
                        labels=['train_error', 'valid_error'],
                        loc='best')
    fig_mslp_error.savefig(config.save_fig_dir + 'mslp_error.png')
    plt.close(fig_mslp_error)

    fig_rmw_error, ax_rmw_error = plt.subplots(figsize=(12, 8))
    train_rmw_error, = plt.plot(np.arange(1, epochs + 1),
                                 train_rmw_error_array[1:], 'r')
    valid_rmw_error, = plt.plot(np.arange(1, epochs + 1),
                                 valid_rmw_error_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('RMW Error')
    plt.title("train/valid error vs epoch")
    ax_rmw_error.legend(handles=[train_rmw_error,
                                 valid_rmw_error],
                        labels=['train_error', 'valid_error'],
                        loc='best')
    fig_rmw_error.savefig(config.save_fig_dir + 'rmw_error.png')
    plt.close(fig_rmw_error)

    fig_r34_error, ax_r34_error = plt.subplots(figsize=(12, 8))
    train_r34_error, = plt.plot(np.arange(1, epochs + 1),
                                 train_r34_error_array[1:], 'r')
    valid_r34_error, = plt.plot(np.arange(1, epochs + 1),
                                 valid_r34_error_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('R34 Error')
    plt.title("train/valid error vs epoch")
    ax_r34_error.legend(handles=[train_r34_error,
                                 valid_r34_error],
                        labels=['train_error', 'valid_error'],
                        loc='best')
    fig_r34_error.savefig(config.save_fig_dir + 'r34_error.png')
    plt.close(fig_r34_error)

if __name__ == '__main__':
    Set_seed.setup_seed(0)
    model = IDOL().to(config.device)

    Estim_Loss = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    w_params = [1, 1, 1, 1]
    train(model, Estim_Loss, optimizer, w_params, config.epochs)

