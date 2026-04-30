import torch.nn as nn
import torch.nn.functional as F
from pointnet.pointnetpp_util import PointNetSetAbstractionMsg, PointNetSetAbstraction

class PointNet2Cls(nn.Module):
    def __init__(self, num_class, normal_channel=False):
        super(PointNet2Cls, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel

        # Set Abstraction Layer 1: 채널 크기 및 nsample 줄이기
        self.sa1 = PointNetSetAbstractionMsg(
            256,  # 샘플링 포인트 수 줄이기: 기존 512 -> 256
            [0.1, 0.2, 0.4], 
            [8, 16, 32],  # nsample 줄이기: 기존 16, 32, 64 -> 8, 16, 32
            in_channel,
            [[16, 16, 32], [32, 32, 64], [32, 48, 64]]  # 채널 크기 절반으로 줄이기
        )

        # Set Abstraction Layer 2: 채널 크기 및 nsample 줄이기
        self.sa2 = PointNetSetAbstractionMsg(
            64,  # 샘플링 포인트 수 줄이기: 기존 128 -> 64
            [0.2, 0.4, 0.8], 
            [8, 16, 32],  # nsample 줄이기: 기존 16, 32, 64 -> 8, 16, 32
            160,  # 이전 레이어의 출력 크기 맞추기
            [[32, 32, 64], [64, 64, 128], [64, 64, 128]]  # 채널 크기 절반으로 줄이기
        )

        # Set Abstraction Layer 3: 채널 크기 줄이기
        self.sa3 = PointNetSetAbstraction(
            None, None, None, 
            320 + 3,  # 이전 레이어의 출력 크기 맞추기
            [128, 256, 512],  # 채널 크기 절반으로 줄이기
            True
        )

        # Fully Connected 레이어 크기 줄이기
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 512)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_points, _, _, _

    def embed(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 512)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x


# class PointNet2Cls(nn.Module):
#     def __init__(self, num_class, normal_channel=False):
#         super(PointNet2Cls, self).__init__()
#         in_channel = 3 if normal_channel else 0
#         self.normal_channel = normal_channel
        
#         # Set Abstraction 레이어 1의 출력 채널 수 줄이기
#         # 기존: [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
#         self.sa1 = PointNetSetAbstractionMsg(
#             512, 
#             [0.1, 0.2, 0.4], 
#             [8, 16, 32],
#             in_channel,
#             [[16, 16, 32], [32, 32, 64], [32, 48, 64]]  # 채널 크기 절반으로 줄이기
#         )

#         # Set Abstraction 레이어 2의 출력 채널 수 줄이기
#         # 기존: [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
#         self.sa2 = PointNetSetAbstractionMsg(
#             128, 
#             [0.2, 0.4, 0.8], 
#             [8, 16, 32],  # nsample 줄이기: 128 -> 64로 감소
#             160,  # 이전 레이어의 채널 수에 맞추어 조정
#             [[32, 32, 64], [64, 64, 128], [64, 64, 128]]  # 채널 크기 절반으로 줄이기
#         )
    
#         # Set Abstraction 레이어 3의 출력 채널 수 줄이기
#         # 기존: [256, 512, 1024]
#         self.sa3 = PointNetSetAbstraction(
#             None, 
#             None, 
#             None, 
#             320 + 3,  # 이전 레이어의 채널 수에 맞추어 조정
#             [128, 256, 512],  # 채널 크기 절반으로 줄이기
#             True
#         )

#         # Fully Connected 레이어 크기 줄이기
#         self.fc1 = nn.Linear(512, 256)  # 기존: 1024 -> 512
#         self.bn1 = nn.BatchNorm1d(256)  # 기존 크기와 맞춤
#         self.drop1 = nn.Dropout(0.4)

#         self.fc2 = nn.Linear(256, 128)  # 기존: 512 -> 256
#         self.bn2 = nn.BatchNorm1d(128)  # 기존 크기와 맞춤
#         self.drop2 = nn.Dropout(0.5)

#         self.fc3 = nn.Linear(128, num_class)  # 기존: 256 -> 128

#     def forward(self, xyz):
#         B, _, _ = xyz.shape
#         if self.normal_channel:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None
#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         x = l3_points.view(B, 512)  # 기존: B, 1024 -> B, 512로 줄이기
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         x = self.fc3(x)
#         x = F.log_softmax(x, -1)

#         return x, l3_points, _, _

#     def embed(self, xyz):
#         B, _, _ = xyz.shape
#         if self.normal_channel:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None
#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         x = l3_points.view(B, 512)  # 기존: B, 1024 -> B, 512로 줄이기
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         x = self.fc3(x)

#         return x


# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()

#     def forward(self, pred, target, trans_feat):
#         total_loss = F.nll_loss(pred, target)

#         return total_loss

