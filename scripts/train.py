import os
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm

from pointops.functions import pointops
from pointnet.pointnetpp_util import index_points

from point_distill.data.datasets import build_logger, get_dataset
from point_distill.models.factory import get_network
from point_distill.distill.losses import M3DLoss
from point_distill.ops.pc_ops import emd_align_and_merge, get_uniformity_score
from point_distill.ops.training import get_time, evaluate_synset


def parse_args():
    p = argparse.ArgumentParser(description='Point Cloud Dataset Distillation')
    p.add_argument('--dataset', type=str, default='MODELNET40',
                   choices=['MODELNET40', 'MODELNET10', 'scanobjectnn', 'shapenet', 'omni'])
    p.add_argument('--model', type=str, default='PointNet',
                   choices=['PointNet', 'PointNetPlusPlus', 'DGCNN',
                            'PointConvDensityClsSsg', 'PointTransformerCls'])
    p.add_argument('--ipc', type=int, default=1, help='synthetic samples per class')
    p.add_argument('--num_exp', type=int, default=1, help='number of experiments to average')
    p.add_argument('--num_eval', type=int, default=5, help='number of evaluation model runs')
    p.add_argument('--epoch_eval_train', type=int, default=500)
    p.add_argument('--Iteration', type=int, default=2000)
    p.add_argument('--lr_img', type=float, default=10)
    p.add_argument('--lr_net', type=float, default=0.01)
    p.add_argument('--batch_real', type=int, default=8)
    p.add_argument('--batch_train', type=int, default=8)
    p.add_argument('--init', type=str, default='real', choices=['real', 'noise'])
    p.add_argument('--data_path', type=str, default='data')
    p.add_argument('--save_path', type=str, default='result')
    p.add_argument('--npoints', type=int, default=255, help='points per anchor chunk')
    p.add_argument('--origin_npoints', type=int, default=1024)
    p.add_argument('--num_morph', type=int, default=4, help='morphed samples per IPC slot')
    return p.parse_args()


def build_morphed_set(pc_syn, alpha, args, num_classes):
    """Expand anchors into morphed training samples for evaluation."""
    split_syn, split_labels = [], []
    for cls in range(num_classes):
        tmp_syn = []
        alpha_cls = alpha[cls * args.ipc * args.num_morph:(cls + 1) * args.ipc * args.num_morph]

        class_chunks = []
        for i in range(args.ipc):
            chunks = torch.chunk(pc_syn[cls * args.ipc + i], chunks=args.samples, dim=1)
            class_chunks.extend(chunks)
        for ch in class_chunks:
            tmp_syn.append(ch.unsqueeze(0))

        for i in range(args.ipc):
            weights = F.softmax(
                alpha_cls[i * args.num_morph:(i + 1) * args.num_morph], dim=1
            ).unsqueeze(-1).unsqueeze(-1)
            chunk_stack = torch.cat(tmp_syn).unsqueeze(0)[
                :, args.samples * i:args.samples * (i + 1), :, :
            ]
            tmp_syn.append((chunk_stack * weights).sum(dim=1))

        n_labels = len(class_chunks) + args.num_morph * args.ipc
        split_syn.append(torch.cat(tmp_syn))
        split_labels.append(torch.full((n_labels,), cls, dtype=torch.long))

    return torch.cat(split_syn, dim=0), torch.cat(split_labels)


def main():
    args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.augment = False

    run_name = (f"result/{args.dataset}_IPC{args.ipc}_{args.model}"
                f"_It{args.Iteration}_init{args.init}")
    os.makedirs(run_name, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.data_path, exist_ok=True)

    logger = build_logger('.', run_name)

    origin_npoints, coord_dim, num_classes, dst_train, _, testloader = \
        get_dataset(args, args.dataset, args.data_path, npoints=args.origin_npoints)
    args.num_classes = num_classes

    args.lr_img = 100 if args.ipc <= 10 else args.lr_img * args.ipc
    args.samples = origin_npoints // args.npoints

    # eval every 500 iters
    eval_it_pool = np.arange(500, args.Iteration + 1, 500).tolist()
    logger.info('eval_it_pool: %s', eval_it_pool)

    accs_all_exps = []
    data_save = []

    for exp in range(args.num_exp):
        logger.info('\n================== Exp %d ==================\n', exp)
        logger.info('args: %s', args.__dict__)

        # --- per-class index ---
        pointcloud_all = torch.cat(
            [dst_train[i][0].unsqueeze(0) for i in range(len(dst_train))], dim=0
        ).to(args.device)
        labels_all_list = [dst_train[i][1] for i in range(len(dst_train))]
        indices_class = [[] for _ in range(num_classes)]
        for i, lab in enumerate(labels_all_list):
            indices_class[lab].append(i)

        for c in range(num_classes):
            logger.info('class %d: %d real samples', c, len(indices_class[c]))

        def get_images(c, n):
            idx = np.random.permutation(indices_class[c])[:n]
            return pointcloud_all[idx]

        # --- compute padding size ---
        quan = (args.num_morph if args.num_morph % args.samples == 0
                else args.samples - args.num_morph % args.samples + args.num_morph)
        syn_npoints = origin_npoints - quan

        # --- initialise synthetic data ---
        pointcloud_syn = torch.tanh(
            torch.randn(num_classes * args.ipc, coord_dim, origin_npoints, device=args.device)
        ).detach()[:, :, :syn_npoints]

        if args.init == 'real':
            logger.info('init from real samples')
            for c in range(num_classes):
                real_set = get_images(c, args.ipc * args.samples)
                if real_set.shape[0] < args.ipc * args.samples:
                    real_set = torch.cat(
                        [real_set, get_images(c, args.ipc * args.samples - real_set.shape[0])],
                        dim=0
                    )
                pc = real_set.permute(0, 2, 1).float()
                pc_sampled = index_points(
                    pc, pointops.furthestsampling(pc.contiguous(), args.npoints).long()
                )
                pc_merged = emd_align_and_merge(pc_sampled, args)
                pointcloud_syn[c * args.ipc:(c + 1) * args.ipc] = pc_merged.detach()
        else:
            logger.info('init from random noise')

        pointcloud_syn = nn.Parameter(pointcloud_syn.requires_grad_(True))
        alpha_syn = torch.randn(
            num_classes * args.ipc * args.num_morph, args.samples,
            requires_grad=True, device=args.device
        )

        optimizer_img = torch.optim.SGD([pointcloud_syn], lr=args.lr_img, momentum=0.5)
        optimizer_alpha = torch.optim.SGD([alpha_syn], lr=10, momentum=0.5)
        criterion = nn.CrossEntropyLoss().to(args.device)
        m3d = M3DLoss()

        logger.info('%s training begins', get_time())

        for it in tqdm(range(args.Iteration + 1)):

            # ----------------------------------------------------------------
            # Evaluation
            # ----------------------------------------------------------------
            if it in eval_it_pool:
                accs = []
                time_list = []
                for it_eval in range(args.num_eval):
                    torch.manual_seed(1996 + it_eval)
                    net_eval = get_network(args.model, 3, num_classes).to(args.device)

                    pc_eval_split, lab_eval_split = build_morphed_set(
                        pointcloud_syn.detach(), alpha_syn.detach(), args, num_classes
                    )
                    lab_eval_split = lab_eval_split.to(args.device)

                    _, _, acc_test, _, _, time_train = evaluate_synset(
                        it_eval, net_eval, pc_eval_split, lab_eval_split,
                        testloader, args.batch_train, args
                    )
                    accs.append(acc_test)
                    time_list.append(time_train)

                logger.info('iter=%d  eval %d × %s  mean=%.4f  std=%.4f  time=%.1fs',
                            it, args.num_eval, args.model,
                            np.mean(accs), np.std(accs), np.mean(time_list))
                accs_all_exps.append(np.mean(accs))

            # ----------------------------------------------------------------
            # Distillation step — fReshape_nodup (paper's method)
            # ----------------------------------------------------------------
            net = get_network(args.model, 3, num_classes).to(args.device)
            net.train()
            for p in net.parameters():
                p.requires_grad = False

            # BatchNorm warm-up
            for module in net.modules():
                if 'BatchNorm' in module._get_name():
                    net(torch.cat([get_images(c, 8) for c in range(num_classes)], dim=0))
                    for m in net.modules():
                        if 'BatchNorm' in m._get_name():
                            m.eval()
                    break

            optimizer_img.zero_grad()
            optimizer_alpha.zero_grad()
            total_loss = torch.tensor(0.0).to(args.device)

            for c in range(num_classes):
                # real: FPS-partition into args.samples chunks, extract features
                pc_real = get_images(c, args.batch_real)
                pc_tmp = index_points(
                    pc_real.permute(0, 2, 1),
                    pointops.furthestsampling(
                        pc_real.permute(0, 2, 1).contiguous(), 1024
                    ).long()
                )
                pc_real_div = []
                for s in range(args.samples):
                    pc_real_div.append(pc_tmp[:, :args.npoints, :])
                    pc_tmp = pc_tmp[:, args.npoints:, :]
                    if pc_tmp.shape[1] > 0:
                        pc_tmp = index_points(
                            pc_tmp,
                            pointops.furthestsampling(
                                pc_tmp.contiguous(), pc_tmp.shape[1]
                            ).long()
                        )
                pc_real_div = torch.cat(pc_real_div, dim=0).permute(0, 2, 1)

                with torch.no_grad():
                    _, _, _, _, layers_real = net(pc_real_div)
                sorted_real = torch.sort(
                    torch.abs(layers_real["x_m"]), dim=2, descending=True
                )[0].detach()

                # uniformity-aware penalty weights
                u_div = get_uniformity_score(pc_real_div).reshape(args.samples, -1).mean(dim=-1)
                u_real = get_uniformity_score(pc_real).mean()
                penalty = torch.exp(-1000 * (u_div - u_real) ** 2)

                # synthetic forward
                pc_syn_c = pointcloud_syn[c * args.ipc:(c + 1) * args.ipc]
                alpha_c = alpha_syn[c * args.ipc * args.num_morph:(c + 1) * args.ipc * args.num_morph]

                split_syn = []
                for i in range(args.ipc):
                    for ch in torch.chunk(pc_syn_c[i], chunks=args.samples, dim=1):
                        split_syn.append(ch.unsqueeze(0))
                for i in range(args.ipc):
                    weights = F.softmax(
                        alpha_c[i * args.num_morph:(i + 1) * args.num_morph], dim=1
                    ).unsqueeze(-1).unsqueeze(-1)
                    stack = torch.cat(split_syn).unsqueeze(0)[
                        :, args.samples * i:args.samples * (i + 1), :, :
                    ]
                    split_syn.append((stack * weights).sum(dim=1))

                _, _, _, _, layers_syn = net(torch.cat(split_syn, dim=0))
                sorted_syn = torch.sort(
                    torch.abs(layers_syn["x_m"]), dim=2, descending=True
                )[0]
                del layers_syn
                torch.cuda.empty_cache()

                # per-partition uniformity-weighted M3D loss
                loss1 = torch.tensor(0.0).to(args.device)
                for n in range(args.samples):
                    loss1 += (
                        m3d(
                            sorted_real.reshape(sorted_real.shape[0], -1)[
                                n * args.batch_real:(n + 1) * args.batch_real
                            ],
                            sorted_syn.reshape(sorted_syn.shape[0], -1)
                        )
                        * penalty[n] * 0.2 * 4 / penalty.sum()
                    )

                total_loss += loss1 / 10
                (loss1 / 10).backward()

            optimizer_img.step()
            optimizer_alpha.step()

            if it % 50 == 0:
                logger.info('%s iter=%04d  loss=%.4f', get_time(), it,
                            total_loss.item() / num_classes)

            if it == args.Iteration:
                data_save.append([
                    copy.deepcopy(pointcloud_syn.detach().cpu()),
                    torch.tensor([i for i in range(num_classes)
                                  for _ in range(args.ipc)], dtype=torch.int)
                ])
                torch.save(
                    {'data': data_save, 'accs': accs_all_exps},
                    os.path.join(args.save_path,
                                 f'{args.dataset}_{args.model}_{args.ipc}ipc.pt')
                )

    logger.info('\n==================== Final ====================')
    logger.info('Mean acc over %d exps: %.2f%%  std=%.2f%%',
                args.num_exp, np.mean(accs_all_exps) * 100,
                np.std(accs_all_exps) * 100)


if __name__ == '__main__':
    random.seed(1992)
    np.random.seed(1992)
    torch.manual_seed(1992)
    torch.cuda.manual_seed_all(1992)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main()
