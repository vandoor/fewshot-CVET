import random

import torch
from torch import Tensor
import numpy as np
from typing import List, Tuple, Iterator
from torch.utils.data.sampler import BatchSampler, Sampler

class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label)+1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class RandomSampler():

    def __init__(self, label, n_batch, n_per):
        self.n_batch = n_batch
        self.n_per = n_per
        self.label = np.array(label)
        self.num_label = self.label.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = torch.randperm(self.num_label)[:self.n_per]
            yield batch


class ClassSampler():
    def __init__(self, label, n_per=None):
        self.n_per = n_per
        label = np.array(label)
        self.m_ind = []
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return len(self.m_ind)

    def __iter__(self):
        classes = torch.arange(len(self.m_ind))
        for c in classes:
            l = self.m_ind[int(c)]
            if self.n_per is None:
                pos = torch.randperm(len(l))
            else:
                pos = torch.randperm(len(l))[:self.n_per]
            yield l[pos]


class TaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks.
    At each iteration, it will sample n_way classes, and the sample support and query images from these classes
    """
    def __init__(self, dataset, n_way, n_shot, n_query, n_tasks):
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks
        self.items_per_label = {}
        for item, label in enumerate(dataset.get_labels()):
            if label in self.items_per_label:
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

    def __len__(self)->int:
        return self.n_tasks

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_tasks):
            yield torch.cat(
                [
                    # pylint: disable=not-callable
                    torch.tensor(random.sample(
                            self.items_per_label[label], self.n_shot+self.n_query
                    ))
                    for label in random.sample(self.items_per_label.keys(), self.n_way)
                ]
            ).tolist()

    def episodic_collate_fn(self, input_data:List[Tuple[Tensor, Tensor, int]])\
            -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[int]]:
        true_class_ids = list({x[2] for x in input_data})
        all_images1 = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images1 = all_images1.reshape(
            (self.n_way, self.n_shot+self.n_query, *all_images1.shape[1:])
        )
        all_images2 = torch.cat([x[1].unsqueeze(0) for x in input_data])
        all_images2 = all_images2.reshape(
            (self.n_way, self.n_shot+self.n_query, *all_images2.shape[1:])
        )
        support_images1 = all_images1[:, :self.n_shot].reshape((-1, *all_images1.shape[2:]))
        query_images1 = all_images1[:, self.n_shot:].reshape((-1, *all_images1.shape[2:]))
        support_images2 = all_images2[:, :self.n_shot].reshape((-1, *all_images2.shape[2:]))
        query_images2 = all_images2[:, self.n_shot:].reshape((-1, *all_images2.shape[2:]))

        all_labels = torch.tensor(
            [true_class_ids.index(x[2]) for x in input_data]
        ).reshape((self.n_way, self.n_shot+self.n_query))
        support_labels = all_labels[:, :self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot:].flatten()

        return (
            support_images1,
            support_images2,
            support_labels,
            query_images1,
            query_images2,
            query_labels,
            true_class_ids
        )


