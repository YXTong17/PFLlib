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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data


class clientFGAC(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3
        )
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma
        )

        # 统计每类数据的数量
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        labels = torch.tensor([label for _, label in train_data])
        self.label_counts = torch.bincount(labels, minlength=self.num_classes)
        # log 平滑
        self.label_counts = torch.log(1 + self.label_counts)

        # self.loss = nn.CrossEntropyLoss(label_smoothing=0.001)

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
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time


class clientFGAC_NTD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3
        )
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma
        )

        # 统计每类数据的数量
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        labels = torch.tensor([label for _, label in train_data])
        self.label_counts = torch.bincount(labels, minlength=self.num_classes)
        # log 平滑
        self.label_counts = torch.log(1 + self.label_counts)

        # NTD
        self.beta = args.beta
        self.tau = args.tau
        self.global_model = None
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
        self.global_model = model.eval().requires_grad_(False)

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
                output_g = self.global_model(x)
                loss = self.loss(output, y)
                loss += self._ntd_loss(output, output_g, y) * self.beta
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    # https://github.com/Lee-Gihun/FedNTD/blob/master/algorithms/fedntd/criterion.py#L30
    def _ntd_loss(self, logits, dg_logits, targets):
        """Not-tue Distillation Loss"""

        # Get smoothed local model prediction
        logits = refine_as_not_true(logits, targets, self.num_classes)
        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        # Get smoothed global model prediction
        with torch.no_grad():
            dg_logits = refine_as_not_true(dg_logits, targets, self.num_classes)
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = (self.tau**2) * self.KLDiv(pred_probs, dg_probs)

        return loss


# https://github.com/Lee-Gihun/FedNTD/blob/master/algorithms/fedntd/utils.py#L6
def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits


class clientFGAC_MCD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3
        )
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma
        )

        # 统计每类数据的数量
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        labels = torch.tensor([label for _, label in train_data])
        self.label_counts = torch.bincount(labels, minlength=self.num_classes)
        # log 平滑
        self.label_counts = torch.log(1 + self.label_counts)

        # NTD
        self.beta = args.beta
        self.tau = args.tau
        self.global_model = None
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
        self.global_model = model.eval().requires_grad_(False)

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
                output_g = self.global_model(x)
                loss = self.loss(output, y)
                loss += self._mcd_loss(output, output_g) * self.beta
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def _mcd_loss(self, logits, dg_logits):
        """Missing Classes Distillation Loss"""

        # Get smoothed local model prediction
        logits = self.refine_as_not_true(logits)
        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        # Get smoothed global model prediction
        with torch.no_grad():
            dg_logits = self.refine_as_not_true(dg_logits)
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = (self.tau**2) * self.KLDiv(pred_probs, dg_probs)

        return loss

    def refine_as_not_true(self, logits):
        present_classes = (
            torch.nonzero(self.label_counts > 0).view(-1).to(logits.device)
        )
        all_classes = torch.arange(0, self.num_classes, device=logits.device)
        mask = ~torch.isin(all_classes, present_classes)
        remaining_classes = all_classes[mask]
        remaining_classes = remaining_classes.repeat(logits.size(0), 1)
        refined_logits = torch.gather(logits, 1, remaining_classes)

        return refined_logits


class clientFGAC_Frozen(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3
        )
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma
        )

        # 统计每类数据的数量
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        labels = torch.tensor([label for _, label in train_data])
        self.label_counts = torch.bincount(labels, minlength=self.num_classes)
        self.missing_classes = (self.label_counts == 0).nonzero(as_tuple=True)[0]
        # log 平滑
        self.label_counts = torch.log(1 + self.label_counts)

        self.register_freeze_hooks()

    def register_freeze_hooks(self):
        """
        注册钩子函数: 在 backward 结束后，将 self.missing_classes 这些行的梯度置 0.
        """
        # 如果你的最终分类层是 self.model.head (nn.Linear)
        # 那么可以在 named_parameters() 里找到它的 weight、bias
        for name, param in self.model.named_parameters():
            if "head.weight" in name:
                # 对 weight 的梯度进行“行置0”操作
                def hook_fn_weight(grad):
                    # grad.shape = [num_classes, in_features]
                    grad[self.missing_classes, :] = 0
                    return grad

                param.register_hook(hook_fn_weight)

            if "head.bias" in name:
                # 对 bias 的梯度进行“元素置0”操作
                def hook_fn_bias(grad):
                    # grad.shape = [num_classes]
                    grad[self.missing_classes] = 0
                    return grad

                param.register_hook(hook_fn_bias)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        self.register_freeze_hooks()

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
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time


class clientFGAC_CC(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3
        )
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

                # Cos
                # normalized_weights = F.normalize(self.model.head.weight, dim=1)
                # cos_sim_matrix = normalized_weights @ normalized_weights.t()
                # penalty = (cos_sim_matrix.sum() - self.num_classes) / 2
                # loss += self.alpha * penalty / self.num_pairs

                # L1
                weight_expanded_i = self.model.head.weight.unsqueeze(1)  # [K, 1, d]
                weight_expanded_j = self.model.head.weight.unsqueeze(0)  # [1, K, d]
                distances = torch.norm(
                    weight_expanded_i - weight_expanded_j, p=1, dim=2
                )  # L1
                triu_indices = torch.triu_indices(
                    self.num_classes, self.num_classes, offset=1
                )
                upper_distances = distances[triu_indices[0], triu_indices[1]]
                penalty = F.relu(100 - upper_distances).mean()
                # penalty = -upper_distances.mean()
                loss += self.alpha * penalty

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        print(penalty)

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time
