import math
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

    def calc_map_map_sim(self, q, k, v):
        d = math.sqrt(k.shape[-1])
        # a: HW
        # b: D
        # c: HW
        # n,m: 2Batch_size
        sft = torch.einsum('nab,mbc->nmac', q, k.transpose(1, 2))
        sft = torch.softmax(sft/d, dim=3)
        vba = torch.einsum('nmac,mcb->nmab', sft, v)
        vba = F.normalize(vba, p=2, dim=3)
        vab = vba.transpose(0, 1)
        vv = torch.sum(vba*vab, dim=3)
        sim = torch.mean(vv, dim=2)
        return sim

    def calc_vec_map_sim(self, u, z):
        # z: (2bs, D) normalized
        # u: (2bs, HW, D) normalized
        sim_hw = torch.einsum('nab,mb->nma', u, z)
        return torch.mean(sim_hw, 2)


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
                # print(self.train_step)
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
                g_loss_tm = time.time()
                # print(f'loss_global_s_s time: {(g_loss_tm - forward_tm)*1000} ms')

                bs, HW, DIM = q.shape
                q = torch.cat((q, q_), dim=0)
                k = torch.cat((k, k_), dim=0)
                v = torch.cat((v, v_), dim=0)
                sim1 = self.calc_map_map_sim(q, k, v)
                exp_sim1 = torch.exp(sim1/args.tau2)
                num_list = exp_sim1[:bs, bs:].diag().repeat(2)
                den_list = torch.sum(exp_sim1, dim=1) - exp_sim1.diag()
                loss_map_map_ss = - torch.sum(torch.log(num_list/den_list+1e-4))
                # print('map_map_loss', loss_map_map_ss)
                total_loss += loss_map_map_ss * args.alpha2
                mapmap_tm = time.time()
                # print(f'map_map_loss time: {1000*(mapmap_tm - g_loss_tm) }ms')
                u = F.normalize(u, p=2, dim=2)
                u_ = F.normalize(u_, p=2, dim=2)
                ulist = torch.cat((u, u_), dim=0)
                sim2 = self.calc_vec_map_sim(ulist, zlist)
                exp_sim2 = torch.exp(sim2/args.tau3)
                num_list = exp_sim2[:bs, bs:].diag().repeat(2)
                den_list = torch.sum(exp_sim2, dim=1) - exp_sim2.diag()
                loss_vec_map_ss = -torch.sum(torch.log(num_list/den_list + 1e-4))
                # print('vec_map_loss', loss_vec_map_ss)
                vecmap_tm = time.time()
                # print(f'vecmap loss time :{1000*(vecmap_tm-mapmap_tm)} ms')
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

                # print('loss_global', loss_global_s)
                g_loss_tm = time.time()
                # print(f'global loss time : {(g_loss_tm - vecmap_tm)*1000} ms')
                total_loss += loss_global_s * args.alpha3
                # print('total_loss:', total_loss)
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
                    print('total steps:', self.train_step)
                    vl, va, vap = self.evaluate(self.val_loader)
                    if va >= self.trlog['max_acc']:
                        self.trlog['max_acc'] = va
                        self.trlog['max_acc_interval'] = vap
                        self.trlog['max_acc_epoch'] = self.train_epoch
                        self.save_model('max_acc')

            # self.lr_scheduler.step()
            self.try_evaluate(epoch)
            print(f'ETA:{self.timer.measure()}/{self.timer.measure(self.train_epoch / args.max_epoch)}')

            if self.train_epoch % 1 == 0:
                self.save_model(f'epoch-{self.train_epoch}')

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
