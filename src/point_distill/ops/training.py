import os
import time
import random
import math

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from point_distill.ops.pc_ops import pc_normalize_batch
from point_distill.data.datasets import TensorDataset


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def get_loops(ipc):
    """Return (outer_loop, inner_loop) hyper-parameters for distillation."""
    table = {1: (1, 1), 3: (5, 1), 5: (5, 1), 10: (5, 1),
             20: (20, 25), 30: (30, 20), 40: (40, 15), 50: (50, 10)}
    if ipc not in table:
        raise ValueError(f"loop hyper-parameters not defined for ipc={ipc}")
    return table[ipc]


def seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed_val)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def epoch(mode, dataloader, net, optimizer, criterion, args, aug=False,
          calc_classwise_acc=False):
    """Run one epoch of training or evaluation.

    Returns:
        (loss_avg, acc_avg, acc_test_per_class, predictions_per_sample)
    """
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    num_classes = args.num_classes
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes
    predictions_per_sample = []

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for datum in dataloader:
        img = datum[0].float().to(args.device)
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        output, *_ = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1),
                               lab.cpu().data.numpy()))

        loss_avg += loss.item() * n_b
        acc_avg += acc
        num_exp += n_b

        if calc_classwise_acc:
            _, predicted = torch.max(output, 1)
            for label, prediction in zip(lab, predicted):
                if label == prediction:
                    correct_per_class[label] += 1
                total_per_class[label] += 1
                predictions_per_sample.append((label.item(), prediction.item()))

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    acc_test_per_class = None
    if calc_classwise_acc:
        acc_test_per_class = [
            correct_per_class[i] / total_per_class[i]
            if total_per_class[i] > 0 else 0.0
            for i in range(num_classes)
        ]

    return loss_avg, acc_avg, acc_test_per_class, predictions_per_sample


def evaluate_synset(it_eval, net, images_train, labels_train, testloader,
                    batch, args):
    """Train net from scratch on synthetic data and evaluate on testloader.

    Returns:
        (net, acc_train, best_acc, best_per_class, best_prediction, time_train)
    """
    net = net.to(args.device)
    images_train = pc_normalize_batch(images_train)

    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch // 2 + 1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train.cpu(), labels_train.cpu())
    g = torch.Generator()
    g.manual_seed(0)
    trainloader = torch.utils.data.DataLoader(
        dst_train, batch_size=batch, shuffle=True,
        num_workers=4, worker_init_fn=seed_worker, generator=g
    )

    start = time.time()
    best_acc = 0.0
    best_per_class = None
    best_prediction = None
    loss_train = acc_train = 0.0

    for ep in tqdm(range(Epoch + 1)):
        loss_train, acc_train, _, _ = epoch(
            'train', trainloader, net, optimizer, criterion, args, aug=False
        )
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                        weight_decay=0.0005)

        if ep % 10 == 0 and ep > 490:
            loss_test, acc_test, acc_test_per_class, predictions = epoch(
                'test', testloader, net, optimizer, criterion, args,
                aug=False, calc_classwise_acc=True
            )
            if acc_test > best_acc:
                best_acc = acc_test
                best_per_class = acc_test_per_class
                best_prediction = predictions

    time_train = time.time() - start
    print(f"{get_time()} Evaluate_{it_eval:02d}: epoch={Epoch:04d} "
          f"time={int(time_train)}s train_acc={acc_train:.4f} "
          f"test_acc={best_acc:.4f}")

    return net, acc_train, best_acc, best_per_class, best_prediction, time_train
