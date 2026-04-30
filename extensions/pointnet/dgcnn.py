import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from pointnet.pointnetpp_util import *

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    # feature = (x).permute(0,3,1,2).contiguous()
    return feature

class DGCNN(nn.Module):
    def __init__(self, output_channels=40):
        super(DGCNN, self).__init__()
        self.k = 20
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        if batch_size == 1:
            x = F.leaky_relu(self.linear1(x), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
            x = self.dp1(x)
            x = F.leaky_relu(self.linear2(x), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
            x = self.dp2(x)
            x = self.linear3(x)     
        else:
            x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
            x = self.dp1(x)
            x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
            x = self.dp2(x)
            x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        return x, None, None, None, None

class PointNetMSG(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetMSG, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points, embed=False):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape #([40, 1024, 3])
        # S = self.npoint #512
        # new_xyz = index_points(xyz, farthest_point_sample(xyz, S)) #torch.Size([40, 512, 3])
        S = N
        new_xyz = xyz[:,:,:]
        new_points_list = []
        grouped_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz) #torch.Size([40, 512, 16])
            grouped_xyz = index_points(xyz, group_idx) #torch.Size([40, 512, 16, 3])
            # grouped_xyz -= new_xyz.view(B, S, 1, C) #torch.Size([40, 512, 16, 3])
            x = new_xyz.view(B, S, 1, C).repeat(1,1,K,1)
            grouped_xyz = torch.cat([grouped_xyz - x, x], dim=3)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                # print(f"{i} : {grouped_points.shape}") 
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]  torch.Size([40, 3, 16, 512])
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                #print(grouped_points.shape)
                grouped_points =  F.relu(bn(conv(grouped_points)))
                #torch.cuda.empty_cache()  # 매 루프마다 GPU 메모리 정리

            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)
            grouped_points_list.append(grouped_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        
        if embed:
            return new_xyz, new_points_concat, grouped_points_list
        else:
            return new_xyz, new_points_concat


class LocalNet(nn.Module):
    def __init__(self, output_channels=40):
        super(LocalNet, self).__init__()
        self.k = 20

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32*2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.sa1 = PointNetMSG(
            1024,  # 샘플링 포인트 수 줄이기: 기존 512 -> 256
            [0.1, 0.2], 
            [16, 32],  # nsample 줄이기: 기존 16, 32, 64 -> 8, 16, 32
            3,
            [[16, 32], [32, 64]]  # 채널 크기 절반으로 줄이기
        )

    def forward(self, x):
        batch_size = x.size(0)
        xyzs = x[:,:,:]
        
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1_knn = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1_knn, k=self.k)
        x = self.conv2(x)
        x2_knn = x.max(dim=-1, keepdim=False)[0]

        x_all = torch.cat((x1_knn, x2_knn), dim=1)   # (B, 32+64=96, N)
        l1_xyz, l1_points = self.sa1(xyzs, None)
        # x_all = torch.cat([x_all, l1_points], dim=1)
        # x_all = l1_points
        # x_all = x1_knn


        return x, {"x_m": x_all}, None, None, {"x_m": x_all}
    

class LightDGCNN(nn.Module):
    def __init__(self, output_channels=40):
        super(LightDGCNN, self).__init__()
        self.k = 20

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32*2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(96, 256, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear1 = nn.Linear(512, 256)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x_all = torch.cat((x1, x2), dim=1)   # (B, 32+64=96, N)

        x = self.conv3(x_all)                # (B, 256, N)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)           # (B, 512)

        x = F.leaky_relu(self.linear1(x), negative_slope=0.2)
        x = self.dp1(x)
        x = self.linear2(x)

        return x, {"x_m": x_all}, None, None, None