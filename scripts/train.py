"""Main dataset distillation training script.

Bugs fixed vs. original:
  1. Evaluation loop used `alpha_syn[c * ...]` (outer loop var) instead of
     `alpha_syn[cls * ...]` (inner loop var) — all classes used class-0 alphas.
  2. Dataset condition `if args.dataset == "MODELNET40" or "MODELNET10" or ...`
     always evaluated True; replaced with `if args.dataset in (...)`.
"""

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
from point_distill.models.factory import get_network, get_eval_pool
from point_distill.distill.losses import M3DLoss, pairwise_interpolate_to_length
from point_distill.ops.pc_ops import (
    emd_align_and_merge, get_uniformity_score,
)
from point_distill.ops.training import get_time, evaluate_synset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method', type=str, default='DC')
    p.add_argument('--dataset', type=str, default='MODELNET40')
    p.add_argument('--model', type=str, default='PointNet')
    p.add_argument('--ipc', type=int, default=1)
    p.add_argument('--eval_mode', type=str, default='S')
    p.add_argument('--num_exp', type=int, default=1)
    p.add_argument('--num_eval', type=int, default=1)
    p.add_argument('--epoch_eval_train', type=int, default=500)
    p.add_argument('--Iteration', type=int, default=2000)
    p.add_argument('--lr_img', type=float, default=10)
    p.add_argument('--lr_net', type=float, default=0.01)
    p.add_argument('--batch_real', type=int, default=8)
    p.add_argument('--batch_train', type=int, default=8)
    p.add_argument('--init', type=str, default='real')
    p.add_argument('--data_path', type=str, default='data')
    p.add_argument('--save_path', type=str, default='result')
    p.add_argument('--dis_metric', type=str, default='ours')
    p.add_argument('--addition_setting', type=str, default='None')
    p.add_argument('--mode', type=str, default='fReshape_nodup')
    p.add_argument('--feature_transform', type=int, default=0)
    p.add_argument('--layer_label', type=str, default='fReshape_nodup')
    p.add_argument('--npoints', type=int, default=255)
    p.add_argument('--origin_npoints', type=int, default=1024)
    p.add_argument('--num_morph', type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.augment = False
    args.dc_aug_param = None

    os.makedirs(args.mode, exist_ok=True)
    suffix = f'_{args.addition_setting}' if args.addition_setting != 'None' else ''
    run_name = (f"{args.mode}/LRimg{args.lr_img}_LRnet{args.lr_net}"
                f"_IPC{args.ipc}_Model_{args.model}_It{args.Iteration}"
                f"_Dataset_{args.dataset}_init_{args.init}{suffix}")

    logger = build_logger('.', run_name)
    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    # --- dataset load (bug fix #2: use `in` instead of chained `or`) ---
    if args.dataset in ('MODELNET40', 'MODELNET10', 'scanobjectnn', 'shapenet', 'omni'):
        origin_npoints, coord_dim, num_classes, dst_train, _, testloader = \
            get_dataset(args, args.dataset, args.data_path, npoints=args.origin_npoints)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    args.num_classes = num_classes

    if args.ipc > 10:
        args.lr_img = args.lr_img * args.ipc
    else:
        args.lr_img = 100

    # eval iteration schedule
    if args.ipc <= 10:
        eval_it_pool = (np.arange(500, args.Iteration + 1, 500).tolist()
                        if args.eval_mode in ('S', 'SS', 'SSS') else [args.Iteration])
    else:
        eval_it_pool = (np.arange(5000, args.Iteration + 1, 1000).tolist()
                        if args.eval_mode in ('S', 'SS', 'SSS') else [args.Iteration])
    logger.info('eval_it_pool: %s', eval_it_pool)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = {k: [] for k in model_eval_pool}
    accs_all = []
    data_save = []

    for exp in range(args.num_exp):
        logger.info('\n================== Exp %d ==================\n', exp)
        logger.info('Hyper-parameters: %s', args.__dict__)
        logger.info('Evaluation model pool: %s', model_eval_pool)

        # --- build per-class index ---
        pointcloud_all = torch.cat(
            [dst_train[i][0].unsqueeze(0) for i in range(len(dst_train))], dim=0
        ).to(args.device)
        labels_all_list = [dst_train[i][1] for i in range(len(dst_train))]
        labels_all = torch.tensor(labels_all_list, dtype=torch.long, device=args.device)
        indices_class = [[] for _ in range(num_classes)]
        for i, lab in enumerate(labels_all_list):
            indices_class[lab].append(i)

        for c in range(num_classes):
            logger.info('class %d: %d real samples', c, len(indices_class[c]))

        def get_images(c, n):
            idx = np.random.permutation(indices_class[c])[:n]
            return pointcloud_all[idx]

        # --- synthetic data initialisation ---
        full_points = origin_npoints
        args.samples = full_points // args.npoints

        label_syn = torch.tensor(
            [np.ones(args.ipc) * i for i in range(num_classes)],
            dtype=torch.int, requires_grad=False, device=args.device
        ).view(-1)

        pointcloud_syn = torch.tanh(
            torch.randn(num_classes * args.ipc, coord_dim, origin_npoints,
                        device=args.device)
        ).detach().clone()

        if args.init == 'real':
            logger.info('initialising synthetic data from real samples')
            quan = (args.num_morph if args.num_morph % args.samples == 0
                    else args.samples - args.num_morph % args.samples + args.num_morph)

            for c in range(num_classes):
                real_set = get_images(c, args.ipc * args.samples)
                if real_set.shape[0] < args.ipc * args.samples:
                    extra = get_images(c, args.ipc * args.samples - real_set.shape[0])
                    real_set = torch.cat([real_set, extra], dim=0)
                pc = real_set.permute(0, 2, 1).float()
                pc_sampled = index_points(
                    pc,
                    pointops.furthestsampling(pc.contiguous(), args.npoints).long()
                )
                pc_sampled_sorted = emd_align_and_merge(pc_sampled, args)
                pointcloud_syn[c * args.ipc:(c + 1) * args.ipc, :, :1024 - quan] = \
                    pc_sampled_sorted.detach()

            pointcloud_syn = pointcloud_syn[:, :, :1024 - quan]
        else:
            logger.info('initialising synthetic data from random noise')
            quan = (args.num_morph if args.num_morph % args.samples == 0
                    else args.samples - args.num_morph % args.samples + args.num_morph)
            pointcloud_syn = pointcloud_syn[:, :, :1024 - quan]

        pointcloud_syn = nn.Parameter(pointcloud_syn.requires_grad_(True))

        # save initial point clouds
        pc_div_name = run_name
        os.makedirs(pc_div_name, exist_ok=True)
        for c in range(num_classes):
            label_folder = os.path.join(pc_div_name, f'class_{c}')
            os.makedirs(label_folder, exist_ok=True)
            pc_np = pointcloud_syn.data[c * args.ipc:(c + 1) * args.ipc].cpu().numpy()
            for i, pc_i in enumerate(pc_np):
                np.savetxt(os.path.join(label_folder, f'init_{c}_{i}.txt'),
                           pc_i.T, delimiter=',')

        alpha_syn = torch.randn(
            num_classes * args.ipc * args.num_morph, args.samples,
            requires_grad=True, device=args.device
        )

        optimizer_img = torch.optim.SGD([pointcloud_syn], lr=args.lr_img, momentum=0.5)
        optimizer_alpha = torch.optim.SGD([alpha_syn], lr=10, momentum=0.5)
        criterion = nn.CrossEntropyLoss().to(args.device)
        m3d_criterion = M3DLoss()

        logger.info('%s training begins', get_time())

        for it in tqdm(range(args.Iteration + 1)):

            # ============================================================
            # Evaluation
            # ============================================================
            if it in eval_it_pool:
                new_batch = args.batch_train

                for model_eval in model_eval_pool:
                    logger.info(
                        'Evaluation | model_train=%s model_eval=%s iter=%d',
                        args.model, model_eval, it
                    )
                    args.epoch_eval_train = 500
                    accs = []
                    accs_per_class = []
                    time_train_list = []

                    for it_eval in range(args.num_eval):
                        torch.manual_seed(1996 + it_eval)
                        net_eval = get_network(model_eval, 3, num_classes,
                                               args.feature_transform).to(args.device)
                        pc_syn_eval = copy.deepcopy(pointcloud_syn.detach())
                        alpha_eval = copy.deepcopy(alpha_syn.detach())

                        split_syn = []
                        split_labels = []

                        for cls in range(num_classes):
                            class_chunks = []
                            tmp_syn = []
                            tmp_labels = []

                            # BUG FIX #1: was `c` (outer-scope loop var), now `cls`
                            alpha_syn_full = alpha_eval[
                                cls * args.ipc * args.num_morph:
                                (cls + 1) * args.ipc * args.num_morph
                            ]

                            label_folder = os.path.join(pc_div_name, f'class_{cls}', 'morphed')
                            os.makedirs(label_folder, exist_ok=True)

                            for i in range(args.ipc):
                                pc = pc_syn_eval[cls * args.ipc + i]
                                chunks = torch.chunk(pc, chunks=args.samples, dim=1)
                                class_chunks.extend(chunks)

                            for ch in class_chunks:
                                tmp_syn.append(ch.unsqueeze(0))
                                tmp_labels.append(cls)

                            for i in range(args.ipc):
                                weights = F.softmax(
                                    alpha_syn_full[i * args.num_morph:(i + 1) * args.num_morph],
                                    dim=1
                                ).unsqueeze(-1).unsqueeze(-1)
                                chunk_stack = torch.cat(tmp_syn).unsqueeze(0)[
                                    :, args.samples * i:args.samples * (i + 1), :, :
                                ]
                                morphed = (chunk_stack * weights).sum(dim=1)

                                for m in range(args.num_morph):
                                    fname = os.path.join(
                                        label_folder,
                                        f'iter_{it}_class_{cls}_ipc_{i}_morph_{m}.txt'
                                    )
                                    np.savetxt(fname, morphed[m].cpu().detach().numpy().T,
                                               delimiter=',')
                                tmp_syn.append(morphed)

                            for _ in range(args.num_morph * args.ipc):
                                tmp_labels.append(cls)

                            split_syn.append(torch.cat(tmp_syn))
                            split_labels.append(torch.tensor(tmp_labels))

                        pc_eval_split = torch.cat(split_syn, dim=0)
                        lab_eval_split = torch.cat(split_labels).to(args.device)

                        _, acc_train, acc_test, acc_test_per_class, _, time_train = \
                            evaluate_synset(it_eval, net_eval, pc_eval_split,
                                            lab_eval_split, testloader, new_batch, args)

                        accs.append(acc_test)
                        time_train_list.append(time_train)

                        if len(accs_per_class) == 0:
                            accs_per_class = [[] for _ in range(num_classes)]
                        for ci in range(num_classes):
                            accs_per_class[ci].append(acc_test_per_class[ci])

                    accs_per_class = [np.mean(accs_per_class[ci]) for ci in range(num_classes)]
                    logger.info(
                        'Evaluate %d random %s  mean=%.4f  std=%.4f  time=%.4f',
                        len(accs), model_eval, np.mean(accs), np.std(accs),
                        np.mean(time_train_list)
                    )
                    accs_all.append(np.mean(accs))
                    if it == args.Iteration:
                        accs_all_exps[model_eval] += accs

                # save checkpoint point clouds
                pc_vis = copy.deepcopy(pointcloud_syn.detach().cpu().numpy())
                for c in range(num_classes):
                    label_folder = os.path.join(pc_div_name, f'class_{c}')
                    os.makedirs(label_folder, exist_ok=True)
                    pc_c = pc_vis[c * args.ipc:(c + 1) * args.ipc]
                    npoints = pc_c.shape[-1] // args.samples
                    for i in range(args.ipc):
                        for s in range(args.samples):
                            chunk = pc_c[i, :, s * npoints:(s + 1) * npoints]
                            np.savetxt(
                                os.path.join(label_folder,
                                             f'iter_{it}_class_{c}_ipc_{i}_sample_{s}.txt'),
                                chunk.T, delimiter=','
                            )

            # ============================================================
            # Training step
            # ============================================================
            net = get_network(args.model, 3, num_classes, args.feature_transform).to(args.device)
            net.train()
            for p in net.parameters():
                p.requires_grad = False

            # BatchNorm warm-up
            for module in net.modules():
                if 'BatchNorm' in module._get_name():
                    pc_real_bn = torch.cat(
                        [get_images(c, 8) for c in range(num_classes)], dim=0
                    )
                    net.train()
                    net(pc_real_bn)
                    for m in net.modules():
                        if 'BatchNorm' in m._get_name():
                            m.eval()
                    break

            loss = torch.tensor(0.0).to(args.device)
            loss_list = []

            optimizer_img.zero_grad()
            optimizer_alpha.zero_grad()

            for c in range(num_classes):
                if args.layer_label in ('fReshape_nodup', 'fReshape_nodup_shapenet',
                                        'fReshape_wo_penalty'):
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
                        pc_tmp = index_points(
                            pc_tmp,
                            pointops.furthestsampling(
                                pc_tmp.contiguous(), pc_tmp.shape[1]
                            ).long()
                        )
                    pc_real_div = torch.cat(pc_real_div, dim=0).permute(0, 2, 1)

                    with torch.no_grad():
                        _, _, _, _, layers_real = net(pc_real_div)
                    sorted_real = torch.sort(torch.abs(layers_real["x_m"]),
                                             dim=2, descending=True)[0].detach()

                    uniformity_real_div = get_uniformity_score(pc_real_div).reshape(
                        args.samples, -1).mean(dim=-1)
                    uniformity_real = get_uniformity_score(pc_real).mean()
                    penalty_weight = torch.exp(-1000 * (uniformity_real_div - uniformity_real) ** 2)

                elif args.layer_label == 'DownSample':
                    pc_real = get_images(c, args.batch_real)
                    with torch.no_grad():
                        _, layers_real, _, _, _ = net(pc_real)
                    sorted_real = torch.sort(torch.abs(layers_real["x_m"]),
                                             dim=2, descending=True)[0].detach()
                else:
                    pc_real = get_images(c, args.batch_real)
                    with torch.no_grad():
                        _, _, _, _, layers_real = net(pc_real)
                    sorted_real = torch.sort(torch.abs(layers_real["x_m"]),
                                             dim=2, descending=True)[0].detach()

                # synthetic forward
                pc_syn_full = pointcloud_syn[c * args.ipc:(c + 1) * args.ipc]
                alpha_syn_full = alpha_syn[
                    c * args.ipc * args.num_morph:(c + 1) * args.ipc * args.num_morph
                ]

                split_syn = []
                class_chunks = []
                for i in range(args.ipc):
                    chunks = torch.chunk(pc_syn_full[i], chunks=args.samples, dim=1)
                    class_chunks.extend(chunks)
                for ch in class_chunks:
                    split_syn.append(ch.unsqueeze(0))

                for i in range(args.ipc):
                    weights = F.softmax(
                        alpha_syn_full[i * args.num_morph:(i + 1) * args.num_morph],
                        dim=1
                    ).unsqueeze(-1).unsqueeze(-1)
                    chunk_stack = torch.cat(split_syn).unsqueeze(0)[
                        :, args.samples * i:args.samples * (i + 1), :, :
                    ]
                    morphed = (chunk_stack * weights).sum(dim=1)
                    split_syn.append(morphed)

                reshaped_pc_syn = torch.cat(split_syn, dim=0)
                _, _, _, _, layers_syn = net(reshaped_pc_syn)
                sorted_syn = torch.sort(torch.abs(layers_syn["x_m"]),
                                        dim=2, descending=True)[0]
                del layers_syn
                torch.cuda.empty_cache()

                # compute loss
                if args.layer_label == 'fReshape_nodup':
                    loss1 = torch.tensor(0.0).to(args.device)
                    for n in range(args.samples):
                        loss1 += (
                            m3d_criterion(
                                sorted_real.reshape(sorted_real.shape[0], -1)[
                                    n * args.batch_real:(n + 1) * args.batch_real
                                ],
                                sorted_syn.reshape(sorted_syn.shape[0], -1)
                            )
                            * penalty_weight[n] * 0.2 * 4 / penalty_weight.sum()
                        )

                elif args.layer_label == 'fReshape_nodup_shapenet':
                    loss1 = torch.tensor(0.0).to(args.device)
                    syn_flat = sorted_syn.reshape(sorted_syn.shape[0], -1)
                    for n in range(args.samples):
                        real_chunk = sorted_real[
                            n * args.batch_real:(n + 1) * args.batch_real
                        ].reshape(args.batch_real, -1)
                        loss_list.append(
                            m3d_criterion(real_chunk, syn_flat)
                            * penalty_weight[n] * 0.2 * 0.25
                        )

                elif args.layer_label == 'fReshape_wo_penalty':
                    loss1 = m3d_criterion(
                        sorted_real.reshape(sorted_real.shape[0], -1),
                        sorted_syn.reshape(sorted_syn.shape[0], -1)
                    ) * 0.2

                elif args.layer_label == 'DownSample':
                    loss1 = m3d_criterion(
                        sorted_real.reshape(sorted_real.shape[0], -1),
                        sorted_syn.reshape(sorted_syn.shape[0], -1)
                    ) * 0.2

                elif args.layer_label == 'Interpolate':
                    interp = pairwise_interpolate_to_length(sorted_syn, sorted_real.shape[-1])
                    loss1 = m3d_criterion(
                        sorted_real.reshape(sorted_real.shape[0], -1),
                        interp.reshape(sorted_syn.shape[0], -1)
                    ) * 0.2

                else:
                    raise ValueError(f"Unknown layer_label: {args.layer_label}")

                if args.layer_label != 'fReshape_nodup_shapenet':
                    loss += loss1 / 10
                    (loss1 / 10).backward()

            if args.layer_label == 'fReshape_nodup_shapenet':
                loss = torch.stack(loss_list).sum() * 0.1
                loss.backward()

            optimizer_img.step()
            optimizer_alpha.step()

            loss_avg = loss.item() / num_classes

            if it % 50 == 0:
                logger.info('%s iter=%04d  loss_avg=%.4f', get_time(), it, loss_avg)

            if it == args.Iteration:
                data_save.append([
                    copy.deepcopy(pointcloud_syn.detach().cpu()),
                    copy.deepcopy(label_syn.detach().cpu())
                ])
                torch.save(
                    {'data': data_save, 'accs_all_exps': accs_all_exps},
                    os.path.join(args.save_path,
                                 f'res_{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc.pt')
                )

    # final summary
    logger.info('\n==================== Final Results ====================')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        logger.info(
            'Run %d experiments, train on %s, evaluate %d random %s: '
            'mean=%.2f%%  std=%.2f%%',
            args.num_exp, args.model, len(accs), key,
            np.mean(accs) * 100, np.std(accs) * 100
        )
    arr = np.array(accs_all).reshape(-1, len(model_eval_pool))
    logger.info('Best row:\n%s', arr[np.argmax(arr.mean(axis=1))])
    logger.info('All rows:\n%s', arr)


if __name__ == '__main__':
    random.seed(1992)
    np.random.seed(1992)
    torch.manual_seed(1992)
    torch.cuda.manual_seed(1992)
    torch.cuda.manual_seed_all(1992)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main()
