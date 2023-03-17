import time
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from model.trainer.base import Trainer
from model.trainer.helper import get_dataloader, prepare_optimizer, prepare_model
from utils import Averager, count_acc, compute_confidence_interval
from tqdm import tqdm


class PreTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model = prepare_model(args)
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
        eye = torch.eye(args.num_classes).cuda()
        for epoch in range(1, args.max_epoch+1):
            self.train_epoch += 1
            self.model.train()
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()
            start_tm = time.time()
            z_list, inner_z_prod = [], []
            q_list, k_list, v_list = [], [], []
            for batch in self.train_loader:
                self.train_step += 1
                if torch.cuda.is_available():
                    data1, data2, gt_label = [_.cuda() for _ in batch]
                else:
                    data1, data2, gt_label = batch[0], batch[1], batch[2]
                data_tm = time.time()
                self.dt.add(data_tm-start_tm)
                results1 = self.model(data1, get_feature=False, require_losses=True)
                results2 = self.model(data2, get_feature=False, require_losses=True)
                logits, q, k, v, z = results1['logits'], results1['q'], results1['k'], results1['v'], results1['z']
                logits_, q_, k_, v_, z_ = results2['logits'], results2['q'], results2['k'], results2['v'], results2['z']

                labels = eye[gt_label].view(len(batch[2]), -1)
                loss_CE = F.cross_entropy(logits, labels)
                tl2.add(loss_CE)

                total_loss = loss_CE
                forward_tm = time.time()
                self.ft.add(forward_tm-data_tm)
                acc = count_acc(logits, gt_label)
                tl1.add(total_loss.item())
                ta.add(acc)

                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm-forward_tm)

                self.optimizer.step()
                optimize_tm = time.time()
                self.ot.add(optimize_tm - backward_tm)

                start_tm = time.time()

            self.lr_scheduler.step()
            self.try_evaluate(epoch)
            print(f'ETA:{self.timer.measure()}/{self.timer.measure(self.train_epoch/args.max_epoch)}')

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        args = self.args
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2))
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = torch.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']
        ))
        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits = self.model(data)['logits']
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        return vl, va, vap

    def evaluate_test(self, data_loader):
        args = self.args
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path,'max_acc.pth'))['params'])
        self.model.eval()
        record = np.zeros((10000, 2))
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']
        ))
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                logits = self.model(data)['logits']
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl
        print('best epoch {}, best val acc = {:.4f} + {:.4f}\n'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']
        ))
        print('Test acc={:.4f} + {:.4f}\n'.format(
            self.trlog['test_acc'],
            self.trlog['test_acc_interval']
        ))
        return vl, va, vap

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

