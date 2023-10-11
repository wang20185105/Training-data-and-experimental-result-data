# -*- coding: utf-8 -*-
"""
@ project: S2AC
@ author: lzx
@ file: model.py
@ time: 2023/10/10 11:06
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

import numba
from math import sqrt
from hausdorff import hausdorff_distance
from transformer.Models import Transformer
import matplotlib.pyplot as plt

__all__ = ['CORAL','AlexNet','Deep_coral','LOG_CORAL']
def CORAL(src,tgt):
    d = src.size(1)
    src_c = coral(src)
    tgt_c = coral(tgt)
    loss = torch.sum(torch.mul((src_c-tgt_c),(src_c-tgt_c)))
    loss = loss/(4*d*d)
    return loss
#EMD distance
def n_discrepancy(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    pre_s, pre_t = F.softmax(y_s, dim=1), F.softmax(y_t, dim=1)
    loss = (-torch.norm(pre_t, 'nuc') + torch.norm(pre_s, 'nuc')) / y_t.shape[0]
    return loss

def CORALAttn(src,tgt):

    d = src.size(1)
    src_c = coral(src)
    tgt_c = coral(tgt)
    term1 = torch.sum(torch.abs(src_c-tgt_c)) / (2 * d)
    term2 = torch.sum(torch.mul(torch.mul((src_c - tgt_c), (src_c - tgt_c)), (src_c - tgt_c))) / (8 * d * d * d)
    term3 = torch.sum(torch.mul(torch.mul((src_c - tgt_c), (src_c - tgt_c)), torch.mul((src_c - tgt_c), (src_c - tgt_c)))) / (16 * d * d * d * d)
    loss = torch.sum(torch.mul((src_c-tgt_c),(src_c-tgt_c)))
    term4 = torch.sum(torch.mul(torch.mul(torch.mul((src_c - tgt_c), (src_c - tgt_c)), torch.mul((src_c - tgt_c), (src_c - tgt_c))), (src_c - tgt_c)))/ (32* d * d * d * d * d)
    loss = loss/(4*d*d)
    term5 = torch.norm(src_c - tgt_c, p=3)
    return loss

def CORALAttn3(src,tgt):

    d = src.size(1)
    src_c = coral(src)
    tgt_c = coral(tgt)
    term1 = torch.sum(torch.abs(src_c-tgt_c)) / (2 * d)
    term2 = torch.sum(torch.mul(torch.mul((src_c - tgt_c), (src_c - tgt_c)), (src_c - tgt_c))) / (8 * d * d * d)
    term3 = torch.sum(torch.mul(torch.mul((src_c - tgt_c), (src_c - tgt_c)), torch.mul((src_c - tgt_c), (src_c - tgt_c)))) / (16 * d * d * d * d)
    loss = torch.sum(torch.mul((src_c-tgt_c),(src_c-tgt_c)))
    term4 = torch.sum(torch.mul(torch.mul(torch.mul((src_c - tgt_c), (src_c - tgt_c)), torch.mul((src_c - tgt_c), (src_c - tgt_c))), (src_c - tgt_c)))/ (32* d * d * d * d * d)
    loss = loss/(4*d*d)
    term5 = torch.norm(src_c - tgt_c, p=3)
    return loss+term2

def LOG_CORAL(src,tgt):
    d = src.size(1)
    upper=True
    src_c = coral(src)
    tgt_c = coral(tgt)
    src_vals, src_vecs = torch.symeig(src_c,eigenvectors = True)
    tgt_vals, tgt_vecs = torch.symeig(tgt_c,eigenvectors = True)
    src_cc = torch.mm(src_vecs,torch.mm(torch.diag(torch.log(src_vals)),src_vecs.t()))
    tgt_cc = torch.mm(tgt_vecs,torch.mm(torch.diag(torch.log(tgt_vals)),tgt_vecs.t()))
    term1 = torch.sum(torch.abs(src_cc - tgt_cc)) / (2 * d)
    term2 = torch.sum(torch.mul(torch.mul((src_c - tgt_c), (src_c - tgt_c)), (src_c - tgt_c))) / (8 * d * d * d)
    term3 = torch.sum(torch.mul(torch.mul((src_c - tgt_c), (src_c - tgt_c)), torch.mul((src_c - tgt_c), (src_c - tgt_c)))) / (16 * d * d * d * d)
    loss = torch.sum(torch.mul((src_cc - tgt_cc), (src_cc - tgt_cc)))
    loss = loss / (4 * d * d)
    return loss

def LOG_CORALAttn(src,tgt):
    d = src.size(1)
    src_c = coral(src)
    tgt_c = coral(tgt)
    src_vals, src_vecs = torch.symeig(src_c,eigenvectors = True)
    tgt_vals, tgt_vecs = torch.symeig(tgt_c,eigenvectors = True)
    src_cc = torch.mm(src_vecs,torch.mm(torch.diag(torch.log(src_vals)),src_vecs.t()))
    tgt_cc = torch.mm(tgt_vecs,torch.mm(torch.diag(torch.log(tgt_vals)),tgt_vecs.t()))
    term1 = torch.sum(torch.abs(src_cc - tgt_cc)) / (2 * d)
    term2 = torch.sum(torch.mul(torch.mul((src_cc - tgt_cc), (src_cc - tgt_cc)), (src_cc - tgt_cc))) / (8 * d * d * d)
    term3 = torch.sum(torch.mul(torch.mul((src_c - tgt_c), (src_c - tgt_c)), torch.mul((src_c - tgt_c), (src_c - tgt_c)))) / (16 * d * d * d * d)
    loss = torch.sum(torch.mul((src_cc - tgt_cc), (src_cc - tgt_cc)))
    term4 = torch.sum(torch.mul(loss, term2))
    term5=torch.norm(src_c - tgt_c,float('inf'))
    loss = loss / (4 * d * d)
    return loss

def LOG_CORALAttn3(src,tgt):
    d = src.size(1)
    src_c = coral(src)
    tgt_c = coral(tgt)
    src_vals, src_vecs = torch.symeig(src_c,eigenvectors = True)
    tgt_vals, tgt_vecs = torch.symeig(tgt_c,eigenvectors = True)
    src_cc = torch.mm(src_vecs,torch.mm(torch.diag(torch.log(src_vals)),src_vecs.t()))
    tgt_cc = torch.mm(tgt_vecs,torch.mm(torch.diag(torch.log(tgt_vals)),tgt_vecs.t()))
    term1 = torch.sum(torch.abs(src_cc - tgt_cc)) / (2 * d)
    term2 = torch.sum(torch.mul(torch.mul((src_cc - tgt_cc), (src_cc - tgt_cc)), (src_cc - tgt_cc))) / (8 * d * d * d)
    term3 = torch.sum(torch.mul(torch.mul((src_c - tgt_c), (src_c - tgt_c)), torch.mul((src_c - tgt_c), (src_c - tgt_c)))) / (16 * d * d * d * d)
    loss = torch.sum(torch.mul((src_cc - tgt_cc), (src_cc - tgt_cc)))
    term4 = torch.sum(torch.mul(loss, term2))
    term5=torch.norm(src_c - tgt_c,float('inf'))
    loss = loss / (4 * d * d)
    return loss+term2+term3

def LOG_CORALAttn4(src,tgt):
    d = src.size(1)
    src_c = coral(src)
    tgt_c = coral(tgt)
    src_vals, src_vecs = torch.symeig(src_c,eigenvectors = True)
    tgt_vals, tgt_vecs = torch.symeig(tgt_c,eigenvectors = True)
    src_cc = torch.mm(src_vecs,torch.mm(torch.diag(torch.log(src_vals)),src_vecs.t()))
    tgt_cc = torch.mm(tgt_vecs,torch.mm(torch.diag(torch.log(tgt_vals)),tgt_vecs.t()))
    term1 = torch.sum(torch.abs(src_cc - tgt_cc)) / (2 * d)
    term2 = torch.sum(torch.mul(torch.mul((src_cc - tgt_cc), (src_cc - tgt_cc)), (src_cc - tgt_cc))) / (8 * d * d * d)
    term3 = torch.sum(torch.mul(torch.mul((src_cc - tgt_cc), (src_cc - tgt_cc)), torch.mul((src_cc - tgt_cc), (src_cc - tgt_cc)))) / (16 * d * d * d * d)
    loss = torch.sum(torch.mul((src_cc - tgt_cc), (src_cc - tgt_cc)))
    term4 = torch.sum(torch.mul(loss, term2))
    term5=torch.norm(src_c - tgt_c,float('inf'))
    loss = loss / (4 * d * d)
    return loss+term2+term3

def coral(data):
    n = data.size(0)
    id_row = torch.ones(n).resize(1,n)
    if torch.cuda.is_available():
        id_row = id_row.cuda()
    sum_column = torch.mm(id_row,data)
    mean_column = torch.div(sum_column,n)
    mean_mean = torch.mm(mean_column.t(),mean_column)
    d_d = torch.mm(data.t(),data)
    coral_result = torch.add(d_d,(-1*mean_mean))*1.0/(n-1)
    return coral_result

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  #
        self.tanh=nn.Tanh()
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        p_q1=self.query_conv(x).view(m_batchsize, -1, width * height)
        p_k1=self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out, attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    def forward(self, x):
        x=x + self.pos_table[:, :x.size(1)].clone().detach()
        return x

class Self_Attn1(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, out_dim, ksize=1, sti=1, pad=0, activation='relu'):
        super(Self_Attn1, self).__init__()
        self.chanel_in = in_dim
        self.chanel_out=out_dim
        self.ksize=ksize
        self.sti=sti
        self.pad=pad
        self.activation = activation
        self.conv1=nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(ksize,ksize), stride=(sti,sti), padding=pad)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim // 8, kernel_size=(ksize,ksize), stride=(sti,sti), padding=pad)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim // 8, kernel_size=(ksize,ksize), stride=(sti,sti), padding=pad)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(ksize,ksize), stride=(sti,sti), padding=pad)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.tanh=nn.Tanh()
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        proj_query=self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value=self.value_conv(x)
        m_batchsize, C_q, width_q, height_q = proj_query.size()
        m_batchsize, C_k, width_k, height_k = proj_key.size()
        m_batchsize, C_v, width_v, height_v = proj_value.size()
        x=self.conv1(x)
        proj_query1=proj_query.view(m_batchsize, -1, width_q * height_q).permute(0, 2, 1)  # B X CX(N)
        proj_key1 =proj_key.view(m_batchsize, -1, width_k * height_k)  # B X C x (*W*H)
        p_q1=proj_query.view(m_batchsize, -1, width_q * height_q)
        p_k1=proj_key.view(m_batchsize, -1, width_k * height_k)
        energy = torch.bmm(proj_query1, proj_key1)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = proj_value.view(m_batchsize, -1, width_v * height_v)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, self.chanel_out, width_q, height_q)
        out = self.gamma * out + x
        return out

class Deep_coral(nn.Module):
    def __init__(self,num_classes = 1000):
        super(Deep_coral,self).__init__()
        self.feature = AlexNet()
        print(self.feature)
        self.fc = nn.Linear(4096,num_classes)
        self.fc.weight.data.normal_(0,0.005)# 原论文中设置的初始化方法

    def forward(self,src,tgt):
        src = self.feature(src)
        src = self.fc(src)
        tgt = self.feature(tgt)
        tgt = self.fc(tgt)
        return src,tgt
#实验四model
class Deep_coralwithposition(nn.Module):
    def __init__(self,num_classes = 1000):
        super(Deep_coralwithposition,self).__init__()
        self.feature = AttentionNet()
        self.fc = nn.Linear(4096,num_classes)
        self.fully_connect_gan1 = nn.Linear(4096, 1)
        self.fully_connect_rot1 = nn.Linear(4096, 4)
        self.fully_connect_rot2 = nn.Linear(4096, 4)
        self.fc.weight.data.normal_(0,0.005)# 原论文中设置的初始化方法
        self.sigmoid = nn.Sigmoid()

    def forward(self,src,tgt):
        src = self.feature(src)
        src_out = self.fc(src)
        src_logits = self.fully_connect_rot1(src)
        tgt = self.feature(tgt)
        tgt_out = self.fc(tgt)
        tgt_logits = self.fully_connect_rot1(tgt)
        return src_out,tgt_out,self.sigmoid(src_logits)#,self.sigmoid(tgt_logits)

#实验四model结束
class Deep_coralAttn(nn.Module):
    def __init__(self,num_classes = 1000):
        super(Deep_coralAttn,self).__init__()
        self.feature = AlexNetAttn()
        self.fc = nn.Linear(4096,num_classes)
        self.fc.weight.data.normal_(0,0.005)# 原论文中设置的初始化方法

    def forward(self,src,tgt):
        src = self.feature(src)
        src = self.fc(src)
        tgt = self.feature(tgt)
        tgt = self.fc(tgt)
        return src,tgt
#source target positional embedding 源,目标都不加
class Deep_coralAttn4(nn.Module):
    def __init__(self,num_classes = 1000):
        super(Deep_coralAttn4,self).__init__()
        self.feature = AttentionNet2()
        self.fc = nn.Linear(4096,num_classes)
        self.fully_connect_gan1 = nn.Linear(4096, 1)
        self.fully_connect_rot1 = nn.Linear(4096, 4)
        self.fc.weight.data.normal_(0,0.005)# 原论文中设置的初始化方法
        self.sigmoid = nn.Sigmoid()

    def forward(self,src,tgt):
        src = self.feature(src,2)
        src_out = self.fc(src)
        src_logits=self.fully_connect_rot1(src)
        tgt = self.feature(tgt,2)
        tgt = self.fc(tgt)
        return src_out,tgt,self.sigmoid(src_logits)
#source target positional embedding 源不加  目标加
class Deep_coralAttn3(nn.Module):
    def __init__(self,num_classes = 1000):
        super(Deep_coralAttn3,self).__init__()
        self.feature = AttentionNet2()
        self.fc = nn.Linear(4096,num_classes)
        self.fully_connect_gan1 = nn.Linear(4096, 1)
        self.fully_connect_rot1 = nn.Linear(4096, 4)
        self.fc.weight.data.normal_(0,0.005)# 原论文中设置的初始化方法
        self.sigmoid = nn.Sigmoid()
    def forward(self,src,tgt):
        src = self.feature(src,2)
        src_out = self.fc(src)
        src_logits=self.fully_connect_rot1(src)
        tgt = self.feature(tgt,1)
        tgt = self.fc(tgt)
        return src_out,tgt,self.sigmoid(src_logits)
#source target positional embedding 源加  目标不加
class Deep_coralAttn2(nn.Module):
    def __init__(self,num_classes = 1000):
        super(Deep_coralAttn2,self).__init__()
        self.feature = AttentionNet2()
        self.fc = nn.Linear(4096,num_classes)
        self.fully_connect_gan1 = nn.Linear(4096, 1)
        self.fully_connect_rot1 = nn.Linear(4096, 4)
        self.fc.weight.data.normal_(0,0.005)# 原论文中设置的初始化方法
        self.sigmoid = nn.Sigmoid()

    def forward(self,src,tgt):
        src = self.feature(src,1)
        src_out = self.fc(src)
        src_logits=self.fully_connect_rot1(src)
        tgt = self.feature(tgt,2)
        tgt = self.fc(tgt)
        return src_out,tgt,self.sigmoid(src_logits)
#完整S2AC
class Deep_coralAttn1(nn.Module):
    def __init__(self,num_classes = 1000):
        super(Deep_coralAttn1,self).__init__()
        self.feature = AttentionNet()
        self.norm=nn.BatchNorm2d(3)
        self.fc = nn.Linear(4096,num_classes)
        self.fully_connect_gan1 = nn.Linear(4096, 1)
        self.fully_connect_rot1 = nn.Linear(4096, 4)
        self.fc.weight.data.normal_(0,0.005)# 原论文中设置的初始化方法
        self.sigmoid = nn.Sigmoid()

    def forward(self,src,tgt):
        src = self.feature(src)
        src_out = self.fc(src)
        src_logits=self.fully_connect_rot1(src)
        tgt = self.feature(tgt)
        tgt = self.fc(tgt)
        return src_out,tgt,self.sigmoid(src_logits)
#可视化feature map
class Deep_coralAttnVisualization(nn.Module):
    def __init__(self,num_classes = 1000):
        super(Deep_coralAttnVisualization,self).__init__()
        self.feature = AttentionNetVisualizaion()
        self.fc = nn.Linear(4096,num_classes)
        self.fully_connect_gan1 = nn.Linear(4096, 1)
        self.fully_connect_rot1 = nn.Linear(4096, 4)
        self.fc.weight.data.normal_(0,0.005)# 原论文中设置的初始化方法
        self.sigmoid = nn.Sigmoid()

    def forward(self,src,tgt):
        _,_,src = self.feature(src)
        src_out = self.fc(src)
        src_logits=self.fully_connect_rot1(src)
        #print('src',src.shape)
        _,_,tgt = self.feature(tgt)
        tgt = self.fc(tgt)
        return src_out,tgt,self.sigmoid(src_logits)
#可视化feature map 结束
'''官方Alexnet网络'''
class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features= nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

class AlexNetwithposition(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNetwithposition, self).__init__()
        self.features= nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.position_enc = PositionalEncoding(6, n_position=6)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.position_enc(x)
        #x, p1 = self.attn1(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
class AlexNetAttn(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNetAttn, self).__init__()
        self.features= nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.attn1 = Self_Attn(256, 'relu')
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x1 = self.features(x)
        x, p1 = self.attn1(x1)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
#------AttentionNet-positional embedding------
class AttentionNet2(nn.Module):
    def __init__(self, num_classes=1000):
        super(AttentionNet2, self).__init__()
        self.position_enc = PositionalEncoding(6, n_position=6)
        self.features = nn.Sequential(
            Self_Attn1(3, 64, 11, 4, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Self_Attn1(64, 192, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Self_Attn1(192, 384, 3, 1, 1),
            nn.ReLU(inplace=True),
            Self_Attn1(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            Self_Attn1(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.attn1 = Self_Attn(256, 'relu')
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

    def forward(self, x,flag):
        x = self.features(x)
        if flag==1:
            x = self.position_enc(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


    #-------AttentionNet---------
class AttentionNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AttentionNet, self).__init__()

        self.position_enc = PositionalEncoding(6,n_position=6)
        self.features = nn.Sequential(
            Self_Attn1(3,64,11,4,2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Self_Attn1(64,192,5,1,2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Self_Attn1(192,384,3,1,1),
            nn.ReLU(inplace=True),
            Self_Attn1(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            Self_Attn1(256,256,3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.attn1 = Self_Attn(256, 'relu')
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x=self.features(x)
        x=self.position_enc(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
#visualize feature map
class Deep_coralAttnVisualization(nn.Module):
    def __init__(self,num_classes = 1000):
        super(Deep_coralAttnVisualization,self).__init__()
        self.feature =AttentionNetVisualizaion()
        self.fc = nn.Linear(4096,num_classes)
        self.fully_connect_gan1 = nn.Linear(4096, 1)
        self.fully_connect_rot1 = nn.Linear(4096, 4)
        self.fc.weight.data.normal_(0,0.005)# 原论文中设置的初始化方法
        self.sigmoid = nn.Sigmoid()

    def forward(self,src,tgt):
        src_x1,src_x2,src = self.feature(src)
        src_out = self.fc(src)
        src_logits=self.fully_connect_rot1(src)
        tgt_x1,tgt_x2,tgt = self.feature(tgt)
        tgt_out = self.fc(tgt)
        return src_out,tgt_out,self.sigmoid(src_logits),src_x1,src_x2,tgt_x1,tgt_x2
class AttentionNetVisualizaion(nn.Module):
    def __init__(self, num_classes=1000):
        super(AttentionNetVisualizaion, self).__init__()
        self.position_enc = PositionalEncoding(6,n_position=6)
        self.features = nn.Sequential(
            Self_Attn1(3,64,11,4,2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Self_Attn1(64,192,5,1,2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Self_Attn1(192,384,3,1,1),
            nn.ReLU(inplace=True),
            Self_Attn1(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            Self_Attn1(256,256,3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.attn1 = Self_Attn(256, 'relu')
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1=self.features(x)
        x2=self.position_enc(x1)
        x = x2.view(x2.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x1,x2,x