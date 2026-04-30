from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import random
import copy
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
from pointops.functions import pointops
from pointnet.pointnetpp_util import index_points
import itertools

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))  # F.relu -> self.relu
        x = self.relu(self.bn2(self.conv2(x)))  # F.relu -> self.relu
        x = self.relu(self.bn3(self.conv3(x)))  # F.relu -> self.relu
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.bn4(self.fc1(x)))  # F.relu -> self.relu
        x = self.relu(self.bn5(self.fc2(x)))  # F.relu -> self.relu
        x = self.fc3(x)
        
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

    # def forward(self, x):
    #     batchsize = x.size()[0]
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = F.relu(self.bn2(self.conv2(x)))
    #     x = F.relu(self.bn3(self.conv3(x)))
    #     x = torch.max(x, 2, keepdim=True)[0]
    #     x = x.view(-1, 1024)

    #     x = F.relu(self.bn4(self.fc1(x))) #40 1024
    #     x = F.relu(self.bn5(self.fc2(x)))
    #     x = self.fc3(x)

    #     iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
    #     if x.is_cuda:
    #         iden = iden.cuda()
    #     x = x + iden
    #     x = x.view(-1, 3, 3)
    #     return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat_df(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat_df, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.relu = nn.ReLU()

        self.dfconv = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 8, 1)
        )
        
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.ada_budg = 0

    def forward(self, x):
        n_pts = x.size()[2]

        if x.shape[0] < 50:
            xyzs_trans = x.permute(0,2,1).contiguous()
            knn_idx = pointops.knnquery_heap(16, xyzs_trans, xyzs_trans).long()
            knn_xyzs = index_points(xyzs_trans, knn_idx)
            diff = xyzs_trans.unsqueeze(2) - knn_xyzs
            diff = diff.permute(0,3,2,1)
            feats = self.dfconv(diff)
        
        #input transform
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
 
        x_1 = self.relu(self.bn1(self.conv1(x)))

        trans_feat = None
        
        pointfeat = x_1
        x_2 = self.relu(self.bn2(self.conv2(x_1)))
        x_m = self.bn3(self.conv3(x_2))
        if x.shape[0] < 50:
            knn_feats = index_points(x_m.permute(0,2,1), knn_idx).permute(0,3,2,1)
            knn_feats = torch.max(knn_feats, dim=2)[0]
            knn_feats = torch.sort(knn_feats, dim=2, descending=True)[0]
            local_feats = torch.sort(feats, dim=2, descending=True)[0].reshape(feats.shape[0], -1, feats.shape[-1])
            local_feats = torch.sort(local_feats, dim=2, descending=True)[0]
            global_feats = torch.sort(x_m, dim=2, descending=True)[0]
            matching_feats = [global_feats, local_feats, knn_feats]
        else:
            matching_feats = None

        x = torch.max(x_m, 2, keepdim=True)[0]
        for i in range(x_m.shape[0]):
            self.ada_budg += torch.unique(torch.max(x_m, 2, keepdim=True)[1][i]).shape[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, matching_feats, x_1, x_2, x_m
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, matching_feats, x_1, x_2, x_m
        

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        

    def forward(self, x):
        n_pts = x.size()[2]
        
        #input transform
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
 
        x_1 = F.relu(self.bn1(self.conv1(x)))

        trans = None
        trans_feat = None
        
        pointfeat = x_1
        x_2 = F.relu(self.bn2(self.conv2(x_1)))
        x_m = self.bn3(self.conv3(x_2))
        x = torch.max(x_m, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat, x_1, x_2, x_m
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat, x_1, x_2, x_m
        
class PointNetfeat_morph(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat_morph, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        

    def forward(self, x, lab=None):
        n_pts = x.size()[2]
        
        #input transform
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
 
        x_1 = F.relu(self.bn1(self.conv1(x)))

        trans = None
        trans_feat = None
        
        pointfeat = x_1
        x_2 = F.relu(self.bn2(self.conv2(x_1)))
        x_m = self.bn3(self.conv3(x_2))

        x = torch.max(x_m, 2, keepdim=True)[0]
        x = x.view(-1, 1024)


        if self.global_feat and lab is not None:
            return x, trans, trans_feat, x_1, x_2, x_m, lab
        elif self.global_feat and lab is None:
            return x, trans, trans_feat, x_1, x_2, x_m
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat, x_1, x_2, x_m

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        # self.feat = PointNetfeat_morph(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x, lab=None):
        if lab is not None:
            x_gf, trans, trans_feat, x_1, x_2, x_m, lab = self.feat(x, lab) # 여기 x가 global?
        else:
            x_gf, trans, trans_feat, x_1, x_2, x_m = self.feat(x)
        x = self.relu(self.bn1(self.fc1(x_gf)))
        f1 = x
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        f2 = x
        x = self.fc3(x)
        f3 = x
        return F.log_softmax(x, dim=1), [f3,f2,f1], trans, trans_feat, {'x_gf':x_gf,'x_1':x_1, 'x_2':x_2, 'x_m':x_m,'f3':f3,'f2':f2,'f1':f1, 'lab': lab}

    def embed(self, x):
        x, trans, trans_feat, x_1, x_2, x_m = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        #x = self.fc3(x)
        return x
    
class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat, _, _, _ = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        f1 = x
        x = F.relu(self.bn2(self.conv2(x)))
        f2 = x
        x = F.relu(self.bn3(self.conv3(x)))
        f3 = x
        x = self.conv4(x)
        f4 = x
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, [f4, f3, f2, f1]

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss



class PointNetfeat_mini(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat_mini, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.global_feat = global_feat
        self.relu = nn.ReLU()
        
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]

        x_1 = self.relu(self.bn1(self.conv1(x)))
        
        pointfeat = x_1
        x_m = self.bn2(self.conv2(x_1))
        x = torch.max(x_m, 2, keepdim=True)[0]
        x = x.view(-1, 128)
        if self.global_feat:
            return x, x_1, x_m
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), x_1, x_m

class PointNetCls_mini(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls_mini, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat_mini(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(128, k)

    def forward(self, x):
        x_gf, x_1, x_m = self.feat(x) # 여기 x가 global?
        x = self.fc1(x_gf)
        return F.log_softmax(x, dim=1), None, None, None, None