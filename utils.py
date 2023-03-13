import os
import shutil
import time
import pprint
import torch
import argparse
import numpy as np


def one_hot(indices, depth):
    """
    :param indices: tensor with size= (n_batch, m) or tensor with size= (m)
    :param depth: a scalar, the depth of the one hot dimension
    :return: A one-hot tensor : (n_batch, m, depth) or (m, depth) tensor
    """
    encoded_indices = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda():
        encoded_indices = encoded_indices.cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indices = encoded_indices.scatter_(1, index, 1)
    return encoded_indices


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu: ', x)


def ensure_path(dir_path, scripts_to_save=None):
    if os.path.exists(dir_path):
        if input(f'{dir_path} exists, remove?[Y]/N') != 'N':
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
    else:
        os.mkdir(dir_path)

    print(f'Experiment dir: {dir_path}')
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for src_file in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(src_file))
            print(f'copy {src_file} to {dst_file}')
            if os.path.isdir(src_file):
                shutil.copytree(src_file, dst_file)
            else:
                shutil.copyfile(src_file, dst_file)


class Averager():
    def __init__(self):
        self.n = 0
        self.avg = 0

    def add(self, x):
        self.avg = (self.n * self.avg + x) / (self.n + 1)
        self.n += 1


def count_acc(logits, label):
    prediction = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (prediction == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (prediction == label).type(torch.FloatTensor).mean().item()


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self):
        x = int(time.time() - self.o)
        if x >= 3600.:
            return f'{x // 3600}h{(x % 3600) // 60}m{(x % 60)}s'
        elif x >= 60:
            return f'{x // 60}m{x % 60}s'
        else:
            return f'{x}s'


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def compute_confidence_interval(data):
    a = 1.0 * np.array(data)
    m, std = np.mean(a), np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def postprocess_args(args):
    args.num_classes = args.way
    save_path1 = '-'.join([args.dataset,
                           args.model_class,
                           args.meta_model_class,
                           args.backbone_class,
                           '{:02d}w{:02d}:{:02d}q'.format(args.way, args.shot, args.query)])
    save_path2 = '_'.join([
        str('_'.join(args.step_size.split(','))),
        str(args.gamma),
        'lr{:.2g}mul{:.2g}'.format(args.lr, args.lr_mul),
        str(args.lr_scheduler),
        f'T1{args.temperature}T2{args.temperature2}',
        f'b{args.balance}',
        'ba_sz{:03d}'.format(max(args.way, args.num_classes)*(args.shot+args.query))
    ])
    if args.init_weights is not None:
        save_path1 += '-Pre'
    if args.use_euclidean:
        save_path1 += '-Dis'
    else:
        save_path1 += '-Sim'

    if args.fix_BN:
        save_path2 += "-FBN"
    else:
        save_path2 += "-NoAug"

    if not os.path.exists(os.path.join(args.save_dir, save_path1)):
        os.mkdir(os.path.join(args.save_dir, save_path1))
    args.save_path = os.path.join(args.save_dir, save_path1, save_path2)
    return args


def get_command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_augment', action='store_true', default=False)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=600)
    parser.add_argument('--num_eval_episodes', type=int, default=600)

    parser.add_argument('--model_class', type=str, default='premod')
    parser.add_argument('--meta_model_class', type=str, default='metamod')
    parser.add_argument('--backbone_class', type=str, default='Res12')
    parser.add_argument('--init_weights', type=str, default=None)
    parser.add_argument('--use_euclidean', action='store_true', default=False)
    parser.add_argument('--D', type=int, default=640)
    parser.add_argument('--tau1', type=float, default=0.1)
    parser.add_argument('--tau2', type=float, default=0.1)
    parser.add_argument('--tau3', type=float, default=0.1)
    parser.add_argument('--tau4', type=float, default=0.1)
    parser.add_argument('--tau5', type=float, default=0.1)

    parser.add_argument('--alpha1', type=float, default=1.0)
    parser.add_argument('--alpha2', type=float, default=1.0)
    parser.add_argument('--alpha3', type=float, default=1.0)
    parser.add_argument('--orig_imsize', type=int, default=-1)

    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--eval_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--eval_shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--eval_query', type=int, default=15)

    # optimizer
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--init_lr', type=float,  default=0.1)
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine','step'])

    # StepLR
    parser.add_argument('--pre_step_size', type=int, default=40)
    parser.add_argument('--pre_gamma', typed=float, default=0.5)
    parser.add_argument('--pre_beta', type=float, default=0.01)
    parser.add_argument('--meta_step_size', type=int, default=50)
    parser.add_argument('--meta_gamma', typed=float, default=0.5)
    parser.add_argument('--meta_beta', type=float, default=0.1)


