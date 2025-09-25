import math
import numpy as np
import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import models
import Config as config
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from scipy.spatial.distance import pdist, squareform
import torch.distributions as dist
import scipy.stats as stats
from scipy.stats import gaussian_kde
from scipy.integrate import quad

class vgg_enc(nn.Module):
    def __init__(self, Pre_Train):
        super().__init__()
        self.backbone = models.vgg13_bn(Pre_Train)
        self.backbone.features._modules['0'] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.share_net = self.backbone.features
        self.flatten = nn.Flatten()

    def forward(self, x_ir):
        x = self.share_net(x_ir)  # 512, 4, 4
        return x

class Plt_Dist(nn.Module):
    def __init__(self, in_dim, id_dim):
        super(Plt_Dist, self).__init__()
        self.fc_m = nn.Linear(in_dim, 1)
        self.fc_s = nn.Linear(in_dim, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.id_dim = id_dim
    def forward(self, plt, dist):
        b, d = plt.size()
        m = self.relu(self.fc_m(plt))
        s = self.relu(self.fc_s(plt))
        dist_samp = m + s * dist
        return dist_samp

def Dist(shf, id_dim=16):
    b, c, n = shf.size()
    flatten = nn.Flatten()
    '''b, 512, 16--->b, 16'''
    if torch.isnan(shf).any():
        error_notice()
        raise ValueError("shf contains NaN")
    fm = torch.mean(shf, dim=1)
    fs = torch.std(shf, dim=1)
    fm = torch.clamp(fm, min=-10, max=10)
    fs = torch.clamp(fs, min=1e-6, max=10)
    dist_samp = torch.normal(fm, fs)
    return dist_samp

# Before entering the loop, unknown bias term: self-learning parameter.
class EquFM(nn.Module):
    def __init__(self, in_dim, uk_node_num, tolerance=1e-3):
        super(EquFM, self).__init__()
        self.tolerance = tolerance
        self.ukn_num = uk_node_num
        self.adj = self.create_adj_matrix(uk_node_num)
        self.gcn1 = GCNConv(in_dim, 2 * in_dim)
        self.gcn2 = GCNConv(2 * in_dim, in_dim)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.fc11 = nn.Linear(in_dim * (uk_node_num + 2), in_dim)
        self.fc12 = nn.Linear(in_dim * 2, in_dim)
        self.fc13 = nn.Linear(in_dim * 2, in_dim)
        self.fc14 = nn.Linear(in_dim * (uk_node_num + 2), in_dim)

    def create_adj_matrix(self, num_nodes):
        adj = torch.eye(num_nodes)

      # Set the connection of adjacent nodes
        for i in range(num_nodes):
            if i > 0:
                adj[i, i - 1] = 1
                adj[i - 1, i] = 1
            if i < num_nodes - 1:
                adj[i, i + 1] = 1
                adj[i + 1, i] = 1

        return adj
    def FM(self, x, num_iter): ##[B, 16]
        # memory = torch.zeros_like(x)  # [B, 16]
        H = torch.zeros_like(x)
        C = torch.randn_like(x) * 1e-6
        uk_nodef = nn.Parameter(torch.randn_like(x).unsqueeze(dim=1).repeat(1, self.ukn_num, 1))
        # Random sampling to generate 0/1 matrix
        # adj = torch.bernoulli(torch.sigmoid(self.adjp))
        adj = self.adj
        adj = adj.to(config.device)
        '''b, uk_node_num, id_num'''
        uk_gdata = GraphD_Construt(uk_nodef, adj)
        ukn, uk_eidx = uk_gdata.x.to(config.device), uk_gdata.edge_index.to(config.device)
        # ukn, uk_eidx = uk_gdata.x, uk_gdata.edge_index
        # wadj = adj * self.sigmoid(self.adjw)
        # wuk_gdata = GraphD_Construt(uk_nodef, wadj)
        # wukn, wuk_eidx = wuk_gdata.x.to(config.device), wuk_gdata.edge_index.to(config.device)
        '''b, uk_node_num * id_num'''
        ukf = self.gcn2(self.gcn1(ukn, uk_eidx), uk_eidx)
        # ukf = self.gcn2(self.gcn1(wukn, wuk_eidx), wuk_eidx)
        ukf = self.flatten(ukf)
        for i in range(num_iter):  # T times loop
            '''forget gate'''
            h_old = H
            a_xh = torch.sigmoid(self.fc11(torch.cat([H, x, ukf], dim=1)))
            ca_xh = C*a_xh
            '''input gate'''
            ga = torch.sigmoid(self.fc12(torch.cat((H, x), dim=1)))
            gv = torch.tanh(self.fc13(torch.cat((H, x), dim=1)))
            C = ca_xh + ga*gv
            '''output gate'''
            a_xh1 = torch.sigmoid(self.fc14(torch.cat([H, x, ukf], dim=1)))
            H = a_xh1 * torch.tanh(C)
            if torch.norm(H - h_old) < self.tolerance:
                print("iterations converged really by tolerance, iter = ", i+1)
                break
        if i == num_iter - 1:
            print("iterations converged by num_iter=", i+1)
        return H, i + 1

    def forward(self, x, num_iter): #torch.Size([B, 16])
        B,_ = x.size()
        H, iter = self.FM(x, num_iter)
        return H, iter

class TCAID_pv_nop2v_ori(nn.Module):
    def __init__(self, in_dim, id_dim):
        super(TCAID_pv_nop2v_ori, self).__init__()
        self.distp = Plt_Dist(in_dim, id_dim)
        self.distv = Plt_Dist(in_dim, id_dim)
        self.uk_node_num = 3
        self.id_dim = id_dim
        # self.uknf_rm = nn.Linear(id_dim * 2, id_dim * self.uk_node_num)
        # self.A = nn.Parameter(torch.randn(id_dim))
        self.A = nn.Parameter(torch.randn(1))
        self.F2_rm = nn.Parameter(torch.randn(2))
        self.F3_rm = nn.Parameter(torch.randn(2))
        self.iter_pow1 = EquFM(id_dim, self.uk_node_num)
        
        self.F4_r34 = nn.Parameter(torch.randn(2))
        self.F5_r34 = nn.Parameter(torch.randn(2))

        self.iter_pow2 = EquFM(id_dim, self.uk_node_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, plt, dist, num_iter=20):
        '''
        Args:
            dist: (batch_size, id_dim=16)
            plt: (batch_size, in_dim=2)
            goal: (batch_size, 4)
        Returns:
            4 task identity tokens: (batch_size, 1, id_dim=16)
        '''
        idp = self.distp(plt, dist)
        idv = self.distv(plt, dist)

        b, d = idp.size()
        '''
        Holland part
        rm = [A / (lnF2(v) - lnF3(v))]^pow = C^pow
        r34 = [C / (lnF4(p) - lnF5(p))]^pow = D^pow
        '''
        F2P = self.F2_rm[0] * idv + self.F2_rm[1]
        F3P = self.F3_rm[0] * idv + self.F3_rm[1]
        lnF2p = torch.log(self.sigmoid(F2P))
        lnF3p = torch.log(self.sigmoid(F3P))
        # b, n, dim = idp.size()
        # A = self.A.reshape(1, 1, dim)
        # C = A / (lnF2p - lnF3p)
        C = self.A[0] / (lnF2p - lnF3p)
        '''idr1 = torch.pow(C, self.pow[0])'''
        idr1, iter1 = self.iter_pow1(C, num_iter)

        F4P = self.F4_r34[0] * idp + self.F4_r34[1]
        F5P = self.F5_r34[0] * idp + self.F5_r34[1]
        lnF4p = torch.log(self.sigmoid(F4P))
        lnF5p = torch.log(self.sigmoid(F5P))
        D = C / (lnF4p - lnF5p)
        idr2, iter2 = self.iter_pow2(D, num_iter)
        return idv, idp, idr1, idr2

def initialize_weights(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

class GEnc_GMMDist(nn.Module):
    def __init__(self, in_channels, out_channels, num_components, feature_dim):
        super().__init__()
        self.num_components = num_components
        self.feature_dim = feature_dim
        self.relu = nn.ReLU()
        self.conv_z = GCNConv(in_channels, 2 * out_channels)
        self.alpha_sh = GCNConv(2 * out_channels, 1)
        self.mu = nn.Parameter(torch.randn(num_components, feature_dim))  # mean vector
        self.log_var = nn.Parameter(torch.randn(num_components, feature_dim))  # std vector
    def forward(self, gdata, dist):
        batch_size = dist.size(0)
        z = self.conv_z(gdata.x, gdata.edge_index)
        alpha_sh = F.softmax(self.alpha_sh(z, gdata.edge_index).squeeze(dim=-1), dim=-1)      # (b, num_components)
        var = torch.exp(self.log_var)  # (num_components, feature_dim)
        dist_expanded = dist.unsqueeze(1)  # (batch_size, 1, feature_dim)
        mu_expanded = self.mu.unsqueeze(0)  # (1, num_components, feature_dim)
        var_expanded = var.unsqueeze(0)  # (1, num_components, feature_dim)
        gmm_dist = mu_expanded + var_expanded * dist_expanded # (batch_size, num_components, feature_dim)
        weighted_gmm_dist = torch.sum(alpha_sh.unsqueeze(dim=-1) * gmm_dist, dim=1)    # (batch_size, feature_dim)
        return weighted_gmm_dist

class PRG_SALSTM8(nn.Module):
    def __init__(self, in_channels):
        super(PRG_SALSTM8, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

        self.conv7 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.conv8 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.conv9 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

        self.conv10 = nn.Conv2d(in_channels*2+1, in_channels, kernel_size=1)
        self.conv11 = nn.Conv2d(in_channels*2+1, in_channels, kernel_size=1)
        self.conv12 = nn.Conv2d(in_channels*2+1, in_channels, kernel_size=1)
        self.conv13 = nn.Conv2d(in_channels*2+1, in_channels, kernel_size=1)

    def sa_conv_lstm(self, x, en_1d): ##[T, B, 512, 4, 4] [B,1,4,4]
        memory = torch.zeros_like(x[0])  # [B, 512, 4, 4]
        H = torch.zeros_like(x[0])
        C = torch.randn_like(x[0]) * 1e-6
        for i in range(x.size(0)):  
            a_xh = torch.sigmoid(self.conv10(torch.cat((H, x[i], en_1d), dim=1)))  
            ca_xh = C*a_xh
            ga = torch.sigmoid(self.conv11(torch.cat((H, x[i], en_1d), dim=1)))
            gv = torch.tanh(self.conv12(torch.cat((H, x[i], en_1d), dim=1)))
            C = ca_xh + ga*gv
            a_xh1 = torch.sigmoid(self.conv13(torch.cat((H, x[i], en_1d), dim=1)))
            H = a_xh1*torch.tanh(C)
            memory, H = self.self_attention_memory(memory, H)  # H:torch.Size([B, 1, 16, 16])
        return H

    def self_attention_memory(self, m, h): 
        vh = self.conv1(h)
        kh = self.conv2(h)
        qh = self.conv3(h)

        qh = torch.transpose(qh, 2, 3)
        ah = F.softmax(kh*qh,dim=-1) 
        zh = vh*ah

        km = self.conv4(m)
        vm = self.conv5(m)
        am = F.softmax(qh*km,dim=-1)
        zm = vm*am
        z0 = torch.cat((zh, zm), dim=1)
        z = self.conv6(z0)
        hz = torch.cat((h, z), dim=1)

        ot = torch.sigmoid(self.conv7(hz))  
        gt = torch.tanh(self.conv8(hz))
        it = torch.sigmoid(self.conv9(hz))

        gi = gt*it
        mf = (1-it)*m
        mt = gi+mf
        ht = ot*mt

        return mt,ht

    def forward(self, x, en_1d): 
        B,_,_,_,_ = x.size() 
        x = x.permute(1, 0, 2, 3, 4) 
        H = self.sa_conv_lstm(x, en_1d)
        # flattened_tensor = H.view(B, -1)
        return H

def GraphD_Construt(nodef, adj):
    edge_index = []
    edge_weight = []
    b, n, d = nodef.size()
    for i in range(n):
        for j in range(n):
            if adj[i, j] != 0:
                edge_index.append([i, j])  
                edge_index.append([j, i])  
                edge_weight.append(adj[i, j])
                edge_weight.append(adj[i, j])

    edge_index = torch.tensor(edge_index).t().contiguous()  
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    data = Data(x=nodef, edge_index=edge_index, edge_attr=edge_weight)
    return data

class IDOL(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = vgg_enc(False)
        self.enc2 = vgg_enc(False)

        self.pr_fc = nn.Linear(4, 16)
        self.prg_fus = PRG_SALSTM8(in_channels=512)
        self.opt = GEnc_GMMDist(in_channels=4, out_channels=4, num_components=4, feature_dim=16)

        self.decoup = TCAID_pv_nop2v_ori(in_dim=2, id_dim=16)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.id_dim, nhead=4, dim_feedforward=config.id_dim * 2,
                                                   activation="gelu", dropout=0.1, batch_first=True)
        self.encoder1 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.encoder3 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.encoder4 = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.flatten = nn.Flatten()

        self.output_msw = nn.Sequential(
            nn.Linear(514 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.output_mslp = nn.Sequential(
            nn.Linear(514 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.output_rmw = nn.Sequential(
            nn.Linear(514 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.output_r34 = nn.Sequential(
            nn.Linear(514 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x, plt, pr, shgt):
        f1 = self.enc1(x[:, :4]).unsqueeze(dim=1)      # b, 1, 512, 4, 4
        f2 = self.enc1(x[:, 4:]).unsqueeze(dim=1)      # b, 1, 512, 4, 4
        tf = torch.cat([f1, f2], dim=1)          # b, 2, 512, 4, 4
        b, t, c, h, w = tf.size()
        prf = self.pr_fc(pr).reshape(b, 1, h, w)            # b, 1, 4, 4
        prgf = self.prg_fus(tf, prf).reshape(b, c, h * w)   # b, 512, 16

        dist = Dist(prgf, id_dim=16)
        nodef = prf.squeeze(dim=1)  # b, 4, 4
        adj = torch.tensor([[0, 0, 1, 1],
                            [1, 0, 1, 0],
                            [1, 1, 0, 0],
                            [0, 1, 0, 1]], dtype=torch.float)

        '''GEnc_GMMDist'''
        gdata = GraphD_Construt(nodef, adj).to(config.device)
        shid = self.opt(gdata, dist)

        shid = shid.reshape(b, h * w)

        idv, idp, idr1, idr2 = self.decoup(plt, dist)

        '''b, 16--->b, 1, 16'''
        idv, idp, idr1, idr2 = idv.unsqueeze(dim=1), idp.unsqueeze(dim=1), idr1.unsqueeze(dim=1), idr2.unsqueeze(dim=1)
        shid = shid.unsqueeze(dim=1)
        '''shid, pid, F'''
        f1 = self.encoder1(torch.cat([shid, idv, prgf], dim=1))
        f2 = self.encoder2(torch.cat([shid, idp, prgf], dim=1))
        f3 = self.encoder3(torch.cat([shid, idr1, prgf], dim=1))
        f4 = self.encoder4(torch.cat([shid, idr2, prgf], dim=1))
        
        msw = self.output_msw(f1)
        mslp = self.output_mslp(f2)
        rmw = self.output_rmw(f3)
        r34 = self.output_r34(f4)
        msw, mslp, rmw, r34 = msw[:, 0], mslp[:, 0], rmw[:, 0], r34[:, 0]

        SimRsh_y = self.SimRloss(shid.squeeze(dim=1), shgt)
        print("SimRsh_y loss: ", SimRsh_y.item())
        SimRsp_y = self.SimRloss(idv.squeeze(dim=1), shgt[:, 0].unsqueeze(dim=1)) +\
                   self.SimRloss(idp.squeeze(dim=1), shgt[:, 1].unsqueeze(dim=1)) +\
                   self.SimRloss(idr1.squeeze(dim=1), shgt[:, 2].unsqueeze(dim=1)) +\
                   self.SimRloss(idr2.squeeze(dim=1), shgt[:, 3].unsqueeze(dim=1))
        print("SimRsp_y loss: ", SimRsp_y.item())
        SimR_loss = SimRsh_y + SimRsp_y
        return msw, mslp, rmw, r34, SimR_loss
    def SimRloss(self, m, n):
        sim_m = m @ m.transpose(0,1)
        sim_n = n @ n.transpose(0,1)
        DiffL = nn.L1Loss()
        simrloss = DiffL(sim_m, sim_n)
        return simrloss
    def initialize(self):
        initialize_weights(self)



