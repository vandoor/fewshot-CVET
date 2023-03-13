import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from model.dataloader.Sampler import CategoriesSampler
from model.models.premod import fs_class_model


def get_dataloader(args):
    if args.dataset == 'cifar100':
        from model.dataloader.Cifar100 import Cifar_100 as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    num_device = torch.cuda.device_count()
    # num_episodes = args.episodes_per_epoch*num_device if args.multi_gpu else args.episodes_per_epoch
    num_episodes = args.episodes_per_epoch
    num_workers = args.num_workers
    trainset = Dataset('train', args, augment=not args.no_augment)
    args.num_class = trainset.num_class
    train_sampler = CategoriesSampler(trainset.labels,
    # TODO: does the n_batch equal episodes_per_epoch?
                                      num_episodes,
                                      max(args.way, args.num_class),
    # TODO: per=shot+query
                                      args.shot+args.query
    )
    train_loader = DataLoader(dataset=trainset,
                              num_workers=num_workers,
                              batch_sampler=train_sampler,
                              pin_memory=True)
    valset = Dataset('val',args)
    val_sampler = CategoriesSampler(valset.labels,
                                    args.num_eval_episodes,
                                    args.eval_way,
                                    args.eval_shot+args.eval_query)
    val_loader = DataLoader(dataset=valset,
                            batch_sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)
    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.labels,
                                     10000,  # args.num_eval_episodes
                                     args.eval_way,
                                     args.eval_shot+args.eval_query
                                     )
    test_loader = DataLoader(dataset=testset,
                             batch_sampler=test_sampler,
                             num_workers=args.num_workers,
                             pin_memory=True)
    return train_loader, val_loader, test_loader


def prepare_model(args):
    model = eval(args.model_class)(args)
    # load pre-trained model (without FC weights)
    if args.init_weights is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.init_weights)['params']
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        print(f'{pretrained_dict.keys()} are loaded!')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    para_model = model.to(device)
    return model, para_model


def prepare_optimizer(model, args):
    top_para = [v for k, v in model.named_parameters() if 'encoder' not in k]
    optimizer = optim.SGD(
        [{'params': model.encoder.parameters()},
         {'params': top_para, 'lr': args.init_lr}],
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    if args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=args.max_epoch,
            eta_min=0,
        )
    elif args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=int(args.step_size),
            gamma=args.gamma
        )
    else:
        raise ValueError('No such scheduler')

    return optimizer, lr_scheduler


# TODO: prepare meta-model and meta-optimizer




