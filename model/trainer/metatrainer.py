import time
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from model.trainer.base import Trainer
from model.trainer.helper import get_meta_dataloader, prepare_meta_model, prepare_meta_optimizer
from utils import Averager, count_acc, compute_confidence_interval
from tqdm import tqdm

class MetaTrainer(Trainer):
    def __init__(self, args):
        super(MetaTrainer, self).__init__(args)
        self.train_loader, self.val_loader, self.test_loader = get_meta_dataloader(args)
        self.model = prepare_meta_model(args)
        self.optimizer, self.lr_scheduler = prepare_meta_optimizer(self.model, args)

    def prepare_labels(self):
        args = self.args
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()

    def train(self):
        args = self.args
        for epoch in range(1, args.max_epoch+1):
            self.train_epoch += 1
            self.model.train()
            start_tm = time.time()

            for batch in self.train_loader:
                self.train_step += 1
                print(self.train_step)
                s_img1, s_img2, s_labels, q_img1, q_img2, q_labels, lid_list = batch
                if torch.cuda.is_available():
                    s_img1, s_img2, s_labels = s_img1.cuda(), s_img2.cuda(), s_labels.cuda()
                    q_img1, q_img2, q_labels = q_img1.cuda(), q_img2.cuda(), q_labels.cuda()
                # s : n_way * shot samples
                # q : n_way * query samples
                # labels: n_way * (shot or query)
                # sup_img = torch.cat([s_img1, s_img2], dim=0)
                # qry_img = torch.cat([q_img1, q_img2], dim=0)
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)
                s_result1 = self.model(s_img1, get_feature=True, get_prototype=True, require_losses=True)
                q_result1 = self.model(q_img1, get_feature=True, require_losses=True)
                s_result2 = self.model(s_img2, get_feature=True, get_prototype=True, require_losses=True)
                q_result2 = self.model(q_img2, get_feature=True, require_losses=True)

                qh = [q_result1['h'], q_result2['h']]
                sp = [s_result1['proto'], s_result2['proto']]
                embed = qh[0].shape[-1]
                print('qhshape:', qh[0].shape)
                print('spshape:', sp[0].shape)
                # qh[0] : (n_way*n_query, embed)
                # sp[0] : (n_way, embed)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                loss_meta = torch.tensor(0.0).cuda()
                for i in range(2):
                    query_embed = qh[i].view(args.way, -1, embed)
                    for j in range(2):
                        proto = sp[j]
                        loss_ij = torch.tensor(0.0).cuda()
                        for k in range(args.way):
                            for u in range(args.query):
                                h = query_embed[k][u]
                                # h : (embed)
                                # proto : (n_way, embed)
                                dist = torch.sum((h.unsqueeze(0) - proto)**2, dim=1)
                                den = torch.log(torch.sum(torch.exp(-dist)))
                                num = torch.sum((h-proto[k])**2)
                                # num = torch.log(torch.exp(-num))
                                loss_ij += -(-num - den)
                        loss_meta += loss_ij / (args.way * args.query)
                loss_meta /= 4
                total_loss = loss_meta

                # qz1, qz2 = q_result1['z'], q_result2['z']
                # sz1, sz2 = s_result1['z'], s_result2['z']
                # # (n_way * n_shot, embed)
                # o1 = torch.mean(sz1.view(args.way, -1, embed), dim=1)
                # o2 = torch.mean(sz2.view(args.way, -1, embed), dim=1)
                # protoz = self.model
                print('loss:',total_loss)
                loss_time = time.time()
                self.lt.add(loss_time - forward_tm)
                self.optimizer.zero_grad()
                total_loss.backward()
                back_time = time.time()
                self.bt.add(back_time - loss_time)
                self.optimizer.step()
                opt_time = time.time()
                self.ot.add(opt_time - back_time)

            self.lr_scheduler.step()
            self.try_evaluate(epoch)
            print(f'ETA:{self.timer.measure()}/{self.timer.measure(self.train_epoch / args.max_epoch)}')

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        args = self.args
        self.model.eval()
        record = np.zeros((args.num_eval_episodes,))
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']
        ))
        with torch.no_grad():
            labels = torch.cat([torch.tensor(i).float().repeat(args.eval_query) for i in range(args.eval_way)])
            for i, batch in enumerate(data_loader):
                s_img, s_img2, s_labels, q_img, q_img2, q_labels, lid_list = batch
                if torch.cuda.is_available():
                    s_img, s_img2, s_labels = s_img.cuda(), s_img2.cuda(), s_labels.cuda()
                    q_img, q_img2, q_labels = q_img.cuda(), q_img2.cuda(), q_labels.cuda()

                s_res = self.model(s_img, get_feature=True, require_losses=False, get_prototype=True)
                q_res = self.model(q_img, get_feature=True, require_losses=False)
                proto = s_res['proto'].unsqueeze(0)  # ->(1, n_way, embed)
                qh = q_res['h'].unsqueeze(1)  # ->(n_way*n_query, 1, embed)
                proto = proto.expand(args.eval_way*args.eval_query, *proto.shape[1:])  # ->(n_way*n_query, n_way, embed)
                print('qh shape:', qh.shape)
                logits = -torch.sum((qh-proto)**2, dim=2)  # ->(n_way*n_query, n_way)
                acc = count_acc(logits, labels)
                record[i] = acc
        va, vap = compute_confidence_interval(record[:, 1])
        self.model.train()
        return 0, va, vap

    def evaluate_test(self):
        pass

    def final_record(self):
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

