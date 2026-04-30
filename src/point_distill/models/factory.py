import torch
import torch.nn as nn

from pointnet import pointnet_model, pointnetpp_model, dgcnn, pointconv
from point_transformer import point_transformer


def get_network(model: str, channel: int, num_classes: int, feature_transform: bool = False):
    """Instantiate a point cloud classification backbone by name."""
    if model == 'PointNet':
        net = pointnet_model.PointNetCls(k=num_classes, feature_transform=feature_transform)
    elif model == 'PointNetPlusPlus':
        net = pointnetpp_model.PointNet2Cls(num_classes)
    elif model == 'DGCNN':
        net = dgcnn.DGCNN(num_classes)
    elif model == 'LightDGCNN':
        net = dgcnn.LightDGCNN(num_classes)
    elif model == 'PointConvDensityClsSsg':
        net = pointconv.PointConvDensityClsSsg(num_classes)
    elif model == 'PointTransformerCls':
        net = point_transformer.PointTransformerCls(num_classes)
    else:
        raise ValueError(f"Unknown model: {model}")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    return net.cuda() if torch.cuda.is_available() else net


def get_eval_pool(eval_mode: str, model: str, model_eval: str) -> list:
    """Return the list of architectures to evaluate on."""
    if eval_mode == 'S':
        return [model.replace('BN', '') if 'BN' in model else model]
    elif eval_mode == 'SS':
        return [model]
    elif eval_mode == 'SSS':
        pool = [model, 'PointNetPlusPlus']
        if model != 'PointNet':
            pool.append('PointNet')
        return pool
    elif eval_mode == 'CrossArchi':
        return ['PointTransformerCls', 'PointConvDensityClsSsg', 'DGCNN', 'PointNet', 'PointNetPlusPlus']
    else:
        return [model_eval]
