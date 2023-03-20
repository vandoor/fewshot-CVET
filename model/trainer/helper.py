import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from model.dataloader.Sampler import CategoriesSampler, TaskSampler
from model.models.premod import PreMod
from model.models.metamod import MetaMod


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
    args.total_samples = trainset.__len__()
    # train_sampler = DirectSampler(n_batch=args.episodes_per_epoch, total_idx=args.total_samples, n_per=args.batch_size)
    train_loader = DataLoader(dataset=trainset, num_workers=0, batch_size=args.batch_size, pin_memory=False)

    valset = Dataset('val', args)
    # val_sampler = DirectSampler(n_batch=args.episodes_per_epoch, total_idx=args.total_samples, n_per=args.batch_size)
    val_loader = DataLoader(dataset=valset, num_workers=0, batch_size=args.batch_size, pin_memory=False)

    testset = Dataset('test', args)
    # test_sampler = DirectSampler(n_batch=args.episodes_per_epoch, total_idx=args.total_samples, n_per=args.batch_size)
    test_loader = DataLoader(dataset=testset, num_workers=0, batch_size=args.batch_size, pin_memory=False)
    return train_loader, val_loader, test_loader

    """
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
    """

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
    return model


def prepare_optimizer(model, args):
    top_para = [v for k, v in model.named_parameters() if 'encoder' not in k]
    optimizer = optim.SGD(
        [{'params': model.encoder.parameters()},
         {'params': top_para, 'lr': args.init_lr}],
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        lr=args.init_lr
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


def get_meta_dataloader(args):
    if args.dataset == 'cifar100':
        from model.dataloader.Cifar100 import Cifar_100 as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    num_episodes = args.episodes_per_epoch
    train_set = Dataset('train', args, augment=not args.no_augment)
    train_sampler = TaskSampler(train_set, args.n_way, args.n_shot, args.n_query, num_episodes)
    train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_sampler=train_sampler,
                              pin_memory=True, collate_fn=train_sampler.episodic_collate_fn)

    val_set = Dataset('val', args)
    val_sampler = TaskSampler(val_set, args.n_way, args.n_shot, args.n_query, num_episodes)
    val_loader = DataLoader(dataset=val_set, num_workers=args.num_workers, batch_sampler=val_sampler, pin_memory=True,
                            collate_fn=val_sampler.episodic_collate_fn)

    test_set = Dataset('test', args)
    test_sampler = TaskSampler(test_set, args.n_way, args.n_shot, args.n_query, num_episodes)
    test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_sampler=test_sampler,
                             pin_memory=True, collate_fn=test_sampler.episodic_collate_fn)
    return train_loader, val_loader, test_loader

def prepare_meta_model(args)->MetaMod:
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
    return model

def prepare_meta_optimizer(model, args):
    top_para = [v for k, v in model.named_parameters() if 'encoder' not in k]
    optimizer = optim.SGD(
        [{'params': model.encoder.parameters()},
         {'params': top_para}],
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        lr=args.init_lr
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


