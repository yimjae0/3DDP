import logging
import os
import torch
from torch.utils.data import Dataset

from pointnet import pointnet_dataloader


def build_logger(work_dir, cfgname):
    assert cfgname is not None
    log_path = os.path.join(work_dir, cfgname + '.log')
    logger = logging.getLogger(cfgname)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


# Dataset root paths — override via environment variable or pass data_root explicitly
_DATASET_ROOTS = {
    'MODELNET40':  os.environ.get('MODELNET40_ROOT',  '/root/dataset/ModelNet40'),
    'MODELNET10':  os.environ.get('MODELNET10_ROOT',  '/root/dataset/ModelNet40'),
    'scanobjectnn': os.environ.get('SONN_ROOT',       '/root/dataset/ScanObjectNN/main_split_nobg'),
    'shapenet':    os.environ.get('SHAPENET_ROOT',    '/root/dataset/ShapeNetv2/PointCloud'),
    'omni':        os.environ.get('OMNI_ROOT',        '/root/dataset/OmniObject3D'),
}

_NUM_CLASSES = {
    'MODELNET40': 40,
    'MODELNET10': 10,
    'scanobjectnn': 15,
    'shapenet': 55,
    'omni': 156,
}


def get_dataset(args, dataset: str, data_path: str, npoints: int = 1024):
    """Load a point cloud classification dataset.

    Returns:
        (origin_npoints, coord_dim, num_classes, train_dataset, train_loader, test_loader)
    """
    if dataset not in _NUM_CLASSES:
        raise ValueError(f"Unknown dataset: {dataset}. "
                         f"Valid: {list(_NUM_CLASSES.keys())}")

    num_classes = _NUM_CLASSES[dataset]
    coord_dim = 3
    batch_size_test = 16 if args.eval_mode == 'CrossArchi' else 128

    root = _DATASET_ROOTS[dataset]

    if dataset in ('MODELNET40', 'MODELNET10'):
        train_dataset = pointnet_dataloader.ModelNetDataLoader(
            root=root, split='train', npoints=npoints, num_category=num_classes)
        test_dataset = pointnet_dataloader.ModelNetDataLoader(
            root=root, split='test', npoints=npoints, num_category=num_classes)

    elif dataset == 'scanobjectnn':
        train_dataset = pointnet_dataloader.ScanObjectNNLoader(root, split='train')
        test_dataset  = pointnet_dataloader.ScanObjectNNLoader(root, split='test')

    elif dataset == 'shapenet':
        train_dataset = pointnet_dataloader.ShapeNetLoader(root, split='train')
        test_dataset  = pointnet_dataloader.ShapeNetLoader(root, split='test')

    elif dataset == 'omni':
        train_dataset = pointnet_dataloader.OmniObject3DLoader(root, split='train')
        test_dataset  = pointnet_dataloader.OmniObject3DLoader(root, split='test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_real, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size_test, shuffle=False)

    return npoints, coord_dim, num_classes, train_dataset, train_loader, test_loader


def get_dataset_seg(dataset: str, data_path: str, npoints: int = 2500):
    """Load ShapeNet part-segmentation dataset."""
    num_classes = 16
    coord_dim = 3
    class_choice = [
        'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone',
        'Guitar', 'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug',
        'Pistol', 'Rocket', 'Skateboard', 'Table',
    ]
    root = '/root/dataset/shapenetcore_partanno_segmentation_benchmark_v0/'
    train_dataset = pointnet_dataloader.ShapeNetDatasetSeg(root=root, class_choice=class_choice)
    test_dataset  = pointnet_dataloader.ShapeNetDatasetSeg(root=root, class_choice=class_choice, split='test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=64, shuffle=False)
    return npoints, coord_dim, num_classes, train_dataset, train_loader, test_loader


class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


class TensorDataset_seg(Dataset):
    def __init__(self, images, labels1, labels2):
        self.images  = images.detach().float()
        self.labels1 = labels1.detach().float()
        self.labels2 = labels2.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels1[index], self.labels2[index]

    def __len__(self):
        return self.images.shape[0]
