# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data


class clientCC(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma
        )

        # 统计每类数据的数量
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        labels = torch.tensor([label for _, label in train_data])
        self.label_counts = torch.bincount(labels, minlength=self.num_classes)
        # log 平滑
        self.label_counts = torch.log(1 + self.label_counts)

        self.alpha = args.alpha
        self.num_pairs = (self.num_classes * (self.num_classes - 1)) / 2

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)

                ##### Cos #####
                # normalized_weights = F.normalize(self.model.head.weight, dim=1)
                # cos_sim_matrix = normalized_weights @ normalized_weights.t()
                # penalty = (cos_sim_matrix.sum() - self.num_classes) / 2
                # loss += self.alpha * penalty / self.num_pairs

                ##### L1 #####
                weight_expanded_i = self.model.head.weight.unsqueeze(1)  # [K, 1, d]
                weight_expanded_j = self.model.head.weight.unsqueeze(0)  # [1, K, d]
                distances = torch.norm(
                    weight_expanded_i - weight_expanded_j, p=1, dim=2
                )  # L1
                # distances = torch.norm(
                #     weight_expanded_i - weight_expanded_j, p=2, dim=2
                # )  # L2
                triu_indices = torch.triu_indices(
                    self.num_classes, self.num_classes, offset=1
                )
                upper_distances = distances[triu_indices[0], triu_indices[1]]
                penalty = F.relu(100 - upper_distances).mean()
                # penalty = -upper_distances.mean()
                loss += self.alpha * penalty

                # Dot
                # penalty = self.compute_logsumexp_dot()
                # loss += self.alpha * penalty

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        print(penalty)

        # self.model.cpu()
        # weight = self.model.head.weight
        # protos = self.collect_protos()
        # dot_mat = torch.tensor(weight @ protos.T)
        # # self.plot_similarity(dot_mat, self.train_time_cost["num_rounds"])

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def compute_logsumexp_dot(self, mean=False):
        weight = self.model.head.weight
        # 1) 计算点积矩阵: dot_mat[i, j] = w_i • w_j
        dot_mat = weight @ weight.T  # [K, K]

        # 2) 取上三角（不含对角线）索引
        K = dot_mat.shape[0]
        triu_indices = torch.triu_indices(K, K, offset=1)  # (2, K*(K-1)/2)

        # 3) 收集所有 i<j 的点积
        pairwise_dots = dot_mat[triu_indices[0], triu_indices[1]]  # shape: [K*(K-1)/2]

        # 4) 做 logsumexp
        #   这里直接对所有对儿做 logsumexp，相当于近似“最大点积” + 光滑
        #   如果想带 margin，可在这里做 (pairwise_dots - margin)，再 logsumexp
        # penalty = torch.logsumexp(pairwise_dots, dim=0)

        if mean:
            penalty = (pairwise_dots**2).mean()
        else:
            penalty = (pairwise_dots**2).sum()

        return penalty

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

            tensor_protos = torch.zeros((self.num_classes, 512), device=self.device)
            # Populate the tensor with aggregated protos
            for label, proto_list in protos.items():
                if len(proto_list) > 0:
                    proto = torch.mean(torch.stack(proto_list), dim=0)
                    tensor_protos[label] = proto

        return tensor_protos

    def plot_similarity(self, dot_mat, round_num, save_path="../MNIST"):
        """
        Plot the similarity matrix for visualization.

        Args:
            dot_mat (torch.Tensor): The similarity matrix of shape (num_classes, num_classes).
            round_num (int): The round number to annotate the plot.
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(dot_mat.cpu().numpy(), cmap="viridis", interpolation="nearest")
        plt.colorbar()
        plt.title(f"Similarity Matrix - Round {round_num}")
        plt.xlabel("Prototype Index")
        plt.ylabel("Classifier Weight Index")
        plt.xticks(range(dot_mat.size(0)))
        plt.yticks(range(dot_mat.size(1)))

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
