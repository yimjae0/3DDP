import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
import glob
import torch
import open3d as o3d
import random
import h5py
import json


warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoints, num_category=10, split='train', process_data=True):
        self.root = root
        self.npoints = npoints
        self.process_data = process_data
        self.num_category = num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
            
        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
                    point_set = farthest_point_sample(point_set, self.npoints)

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            point_set = farthest_point_sample(point_set, self.npoints)
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        point_set = point_set[:, 0:3].T

        return torch.from_numpy(point_set), torch.tensor(label[0])

    def __getitem__(self, index):
        return self._get_item(index)
    
def load_h5(h5_filename):
    with h5py.File(h5_filename, 'r', locking=False) as f:
        data = f['data'][:]
        label = f['label'][:]
    return data, label
    
class ScanObjectNNLoader(Dataset):
    def __init__(self, root, split='train'):

        if split == 'train':
            self.dataset, self.label = load_h5(root + "/sampled_train.h5")
        else:
            self.dataset, self.label = load_h5(root + "/sampled_test.h5")

    def __len__(self):
        return len(self.label)

    def _get_item(self, index):
        point_set = self.dataset[index]
        label = self.label[index]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        point_set = point_set[:, 0:3].T #왜????

        return torch.from_numpy(point_set), torch.tensor(label)

    def __getitem__(self, index):
        return self._get_item(index)

class ShapeNetLoader(Dataset):
    def __init__(self, root, split='train'):

        self.label_list = []

        if split == 'train':
            self.data_list = glob.glob(root + '/*/train/*.npy')
            for d in tqdm(self.data_list):
                self.label_list.append(d.split('/')[-3])
        else:
            self.data_list = glob.glob(root + '/*/test/*.npy')
            for d in tqdm(self.data_list):
                self.label_list.append(d.split('/')[-3])

    def __len__(self):
        return len(self.data_list)

    def _get_item(self, index):
        point_path = self.data_list[index]
        label = self.label_list[index]

        point = np.load(point_path)

        return torch.from_numpy(point).T, torch.tensor(int(label))

    def __getitem__(self, index):
        return self._get_item(index)
    

class OmniObject3DLoader(Dataset):
    def __init__(self, root, split='train'):

        self.label_list = []
        self.label_dict = {}
        self.label_path = sorted(glob.glob(root + '/train/1024/*'))[1:]
        for i, label in enumerate(self.label_path):
            label_id = label.split('/')[-1]
            if os.path.isdir(label):
                self.label_dict[label_id] = i

        if split == 'train':
            self.data_list = glob.glob(root + '/train/1024/*/*/*.ply')
            for d in tqdm(self.data_list):
                self.label_list.append(self.label_dict[d.split('/')[-3]])
        else:
            self.data_list = glob.glob(root + '/test/1024/*/*/*.ply')
            for d in tqdm(self.data_list):
                self.label_list.append(self.label_dict[d.split('/')[-3]])

    def __len__(self):
        return len(self.data_list)

    def _get_item(self, index):
        point_path = self.data_list[index]
        label = self.label_list[index]

        pcd = o3d.io.read_point_cloud(point_path)
        point = np.asarray(pcd.points, dtype=np.float32)
        point[:, 0:3] = pc_normalize(point[:, 0:3])

        return torch.from_numpy(point).T, torch.tensor(int(label))

    def __getitem__(self, index):
        return self._get_item(index)

# class TensorDataset(Dataset):
#     def __init__(self, images, labels): # images: n x c x h x w tensor
#         self.images = images.detach().float()
#         self.labels = labels.detach()

#     def __getitem__(self, index):
#         return self.images[index], self.labels[index]

#     def __len__(self):
#         return self.images.shape[0]

class ShapeNetDatasetSeg(Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train'):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.classification = classification
        self.seg_classes = {}

        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for idx, fn in enumerate(self.meta[item]):
                self.datapath.append((item, fn[0], fn[1]))
                # if split == 'train' and idx == 10:
                #     break

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        # self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        self.num_seg_classes = max(self.seg_classes.values())
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        # choice = np.random.choice(len(seg), self.npoints, replace=True)
        if len(seg) < self.npoints:
            choice = np.array(random.sample(range(0, len(seg)), len(seg)))
            choice = np.concatenate([choice, np.random.choice(choice, self.npoints - len(seg), replace=True)])
        else:
            choice = np.array(random.sample(range(0, len(seg)), self.npoints))
        # resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return point_set.T, seg.unsqueeze(0), fn, self.classes[fn[0]]

    def __len__(self):
        return len(self.datapath)
    