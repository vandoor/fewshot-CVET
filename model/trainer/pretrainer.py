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
        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()
            start_tm = time.time()
            for batch in self.train_loader:
                self.train_step += 1
                print(self.train_step)
                if torch.cuda.is_available():
                    data1, data2, gt_label = [_.cuda() for _ in batch]
                else:
                    data1, data2, gt_label = batch[0], batch[1], batch[2]
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)
                results1 = self.model(data1, get_feature=False, require_losses=True)
                results2 = self.model(data2, get_feature=False, require_losses=True)
                logits, q, k, v, z = results1['logits'], results1['q'], results1['k'], results1['v'], results1['z']
                logits_, q_, k_, v_, z_ = results2['logits'], results2['q'], results2['k'], results2['v'], results2['z']
                u, u_ = results1['u'], results2['u']

                labels = eye[gt_label].view(len(batch[2]), -1)
                loss_CE = F.cross_entropy(logits, labels)

                total_loss = loss_CE

                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, gt_label)

                z = F.normalize(z, p=2, dim=1)
                z_ = F.normalize(z_, p=2, dim=1)
                num_list = torch.sum(z * z_, dim=1).repeat(2)
                zlist = torch.cat((z, z_), dim=0)
                den_list = [torch.sum(torch.exp(torch.sum(zlist[i] * zlist / args.tau1, dim=1)))
                            - torch.exp(zlist[i].dot(zlist[i]) / args.tau1)
                            for i in range(len(zlist))]
                den_list = torch.stack(den_list)
                num_list = torch.exp(num_list / args.tau1)
                loss_global_ss = -torch.sum(torch.log(num_list / den_list + 1e-5))
                total_loss += args.alpha1 * loss_global_ss

                # print(f'qkv shape: {q.shape}, {k.shape}, {v.shape}')
                bs, HW, DIM = q.shape
                num_list = self.model.slf_attn.attention(q, k_, v_)[0]
                num_list2 = self.model.slf_attn.attention(q_, k, v)[0]
                num_list = F.normalize(num_list, p=2, dim=2)
                num_list2 = F.normalize(num_list2, p=2, dim=2)
                num_list = (torch.sum(torch.sum(num_list * num_list2, dim=2), dim=1) / HW)
                num_list = (num_list / args.tau2).repeat(2)
                # print("numlist:", num_list)
                den_list = torch.zeros((2 * bs)).cuda()
                q = torch.cat((q, q_), dim=0)
                k = torch.cat((k, k_), dim=0)
                v = torch.cat((v, v_), dim=0)
                for l in range(1, 2 * bs):
                    q_rot = torch.cat((q[l:], q[:l]), dim=0)
                    k_rot = torch.cat((k[l:], k[:l]), dim=0)
                    v_rot = torch.cat((v[l:], v[:l]), dim=0)
                    L = F.normalize(self.model.slf_attn.attention(q_rot, k, v)[0], p=2, dim=2)
                    R = F.normalize(self.model.slf_attn.attention(q, k_rot, v_rot)[0], p=2, dim=2)
                    delta_batch = torch.exp(torch.sum(torch.sum(L * R, dim=2), dim=1) / HW / args.tau2)
                    # print(delta_batch)
                    den_list += delta_batch
                    # print(den)

                # print("----------------")
                # print(num_list)
                # print(den_list)
                loss_map_map_ss = - torch.sum(num_list - torch.log(den_list + 1e-4))
                total_loss += loss_map_map_ss * args.alpha2
                # print(u.shape)

                u = F.normalize(u, p=2, dim=2)
                # print('z:',z.unsqueeze(1).shape)
                # print('zlist:', zlist.shape)
                # print('u:', u.shape)
                num_list = torch.cat((
                    torch.exp(torch.sum(torch.sum(u * z_.unsqueeze(1), dim=2), dim=1) / HW / args.tau3),
                    torch.exp(torch.sum(torch.sum(u_ * z.unsqueeze(1), dim=2), dim=1) / HW / args.tau3)
                ))
                den_list = torch.zeros([2 * bs]).cuda()
                ulist = torch.cat((u, u_), dim=0)
                for l in range(2 * bs):
                    zlist = torch.cat((zlist[1:], zlist[:1]), dim=0)
                    den_list += torch.exp(
                        torch.sum(torch.sum(ulist * zlist.unsqueeze(1), dim=2), dim=1) / HW / args.tau3)

                loss_vec_map_ss = -torch.sum(torch.log(num_list / den_list + 1e-4))
                total_loss += loss_vec_map_ss * args.alpha2
                loss_global_s = 0

                for i in range(2 * bs):
                    idx = torch.arange(bs)[gt_label == gt_label[i % bs]]
                    idx = torch.cat((idx, idx + bs), dim=0)
                    num = torch.sum(torch.exp(torch.sum(zlist[i] * zlist[idx] / args.tau4, dim=1))) - \
                          torch.exp(zlist[i].dot(zlist[i]) / args.tau4)
                    # print(zlist[i])
                    # print(zlist[idx])
                    den = torch.sum(torch.exp(torch.sum(zlist[i] * zlist / args.tau4, dim=1))) - \
                          torch.exp(zlist[i].dot(zlist[i]) / args.tau4)
                    # print(idx)
                    # print("num:", num)
                    # print("den:", den)
                    loss_global_s += -torch.log(num / den + 1e-4) / idx.shape[0]

                print('loss_global', loss_global_s)
                total_loss += loss_global_s * args.alpha3
                print('total_loss:', total_loss)

                tl1.add(total_loss.item())
                ta.add(acc)

                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimize_tm = time.time()
                self.ot.add(optimize_tm - backward_tm)

                start_tm = time.time()
                if self.train_step % args.episodes_per_epoch == 0:
                    break

            self.lr_scheduler.step()
            self.try_evaluate(epoch)
            print(f'ETA:{self.timer.measure()}/{self.timer.measure(self.train_epoch / args.max_epoch)}')

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        args = self.args
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2))
        # label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        # label = torch.type(torch.LongTensor)
        # if torch.cuda.is_available():
        #     label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']
        ))
        eye = torch.eye(args.num_classes).cuda()
        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if i >= len(record):
                    break
                if torch.cuda.is_available():
                    data, data2, gt_label = [_.cuda() for _ in batch]
                else:
                    data, data2, gt_label = batch[0], batch[1], batch[2]

                logits = self.model(data)['logits']
                labels = eye[gt_label]
                loss = F.cross_entropy(logits, labels)
                acc = count_acc(logits, gt_label)
                record[i - 1, 0] = loss.item()
                record[i - 1, 1] = acc
        # assert (i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        return vl, va, vap

    def evaluate_test(self):
        args = self.args
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        record = np.zeros((10000, 2))
        # label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        # label = label.type(torch.LongTensor)
        # if torch.cuda.is_available():
        #     label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']
        ))
        eye = torch.eye(args.num_classes).cuda()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):
                if torch.cuda.is_available():
                    data, data2, gt_label = [_.cuda() for _ in batch]
                else:
                    data, data2, gt_label = batch[0], batch[1], batch[2]
                logits = self.model(data)['logits']
                labels = eye[gt_label]
                loss = F.cross_entropy(logits, gt_label)
                acc = count_acc(logits, labels)
                record[i - 1, 0] = loss.item()
                record[i - 1, 1] = acc
        assert (i == record.shape[0])
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
