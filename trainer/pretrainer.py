import time
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from trainer.base import Trainer
from trainer.helper import get_dataloader, prepare_optimizer, prepare_model
from utils import pprint, ensure_path, Averager, Timer, count_acc, one_hot, compute_confidence_interval
from tensorboardX import SummaryWriter
from collections import deque
from tqdm import tqdm


class PreTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

    def prepare_label(self):
        args = self.args
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        return label

    def train(self):
        # TODO
        args = self.args
        if self.args.fix_BN:
            self.model.encoder.eval()
        label = self.prepare_label()
        for epoch in range(1, args.max_epoch+1):
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()
            start_tm = time.time()
            for batch in self.train_loader:
                print(f'batch = {batch}')
                self.train_step += 1
                if torch.cuda.is_available():
                    data, gt_label = [_.cuda() for _ in batch]
                else:
                    data, gt_label = batch[0], batch[1]



    def evaluate(self, data_loader):
        # TODO
        args = self.args
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2))

    def evaluate_test(self, data_loader):
        # TODO
        pass

    def final_record(self):
        # save the best performance
        with open(osp.join(self.args.save_path, f"{self.trlog['test_acc']}+{self.trlog['test_acc_interval']}")) as f:
            f.write("best epoch {}, best val_acc={:.4f} + {:.4f}\n".format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']
            ))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']
            ))

