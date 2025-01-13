import copy
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from flcore.clients.clientbase import Client


class clientTGCR(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.TGCR = None
        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda

    def set_TGCR(self, TGCR):
        self.TGCR = TGCR

    def train(self):
        trainloader = self.load_train_data()
        # model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.TGCR is not None:
                    loss += self.lamda * self.loss_mse(
                        self.model.head.weight, self.TGCR
                    )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time
