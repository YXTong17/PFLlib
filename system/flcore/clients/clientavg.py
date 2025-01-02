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

import copy
import time

import numpy as np
import torch
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

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


# class clientAVG_Frozen(clientAVG):
#     def __init__(self, args, id, train_samples, test_samples, **kwargs):
#         super().__init__(args, id, train_samples, test_samples, **kwargs)

#         # 统计每类数据的数量
#         train_data = read_client_data(self.dataset, self.id, is_train=True)
#         labels = torch.tensor([label for _, label in train_data])
#         self.label_counts = torch.bincount(labels, minlength=self.num_classes)
#         self.missing_classes = (self.label_counts == 0).nonzero(as_tuple=True)[0]

#         self.register_freeze_hooks()

#     def register_freeze_hooks(self):
#         """
#         注册钩子函数: 在 backward 结束后，将 self.missing_classes 这些行的梯度置 0.
#         """
#         # 如果你的最终分类层是 self.model.head (nn.Linear)
#         # 那么可以在 named_parameters() 里找到它的 weight、bias
#         for name, param in self.model.named_parameters():
#             if "head.weight" in name:
#                 # 对 weight 的梯度进行“行置0”操作
#                 def hook_fn_weight(grad):
#                     # grad.shape = [num_classes, in_features]
#                     grad[self.missing_classes, :] = 0
#                     return grad

#                 param.register_hook(hook_fn_weight)

#             if "head.bias" in name:
#                 # 对 bias 的梯度进行“元素置0”操作
#                 def hook_fn_bias(grad):
#                     # grad.shape = [num_classes]
#                     grad[self.missing_classes] = 0
#                     return grad

#                 param.register_hook(hook_fn_bias)


class clientAVG_Frozen(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # 统计每类数据的数量
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        labels = torch.tensor([label for _, label in train_data])
        self.label_counts = torch.bincount(labels, minlength=self.num_classes)
        self.missing_classes = (self.label_counts == 0).nonzero(as_tuple=True)[0]

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

                # Manually zero out the gradients for missing classes
                for name, param in self.model.head.named_parameters():
                    if "weight" in name:
                        param.grad[self.missing_classes, :] = 0
                    if "bias" in name:
                        param.grad[self.missing_classes] = 0

                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time
