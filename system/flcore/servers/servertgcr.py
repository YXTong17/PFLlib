import time
from collections import defaultdict
from threading import Thread

import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clienttgcr import clientTGCR
from flcore.servers.serverbase import Server
from torch.utils.data import DataLoader


class FedTGCR(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientTGCR)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.server_learning_rate = args.local_learning_rate
        self.batch_size = args.batch_size
        self.server_epochs = args.server_epochs

        self.feature_dim = args.feature_dim
        self.server_hidden_dim = self.feature_dim

        self.TGCR = Trainable_Global_Classifier_Regularizer(
            self.num_classes, self.server_hidden_dim, self.feature_dim, self.device
        ).to(self.device)
        print(self.TGCR)
        self.CEloss = nn.CrossEntropyLoss()

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate fine-tuned local model")
                self.evaluate()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

            self.receive_classifiers()
            self.update_TGCR()
            self.send_TGCR()

            self.Budget.append(time.time() - s_t)
            print("-" * 50, self.Budget[-1])

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

    def send_TGCR(self):
        assert len(self.clients) > 0

        TGCR = self.TGCR(torch.arange(self.num_classes, device=self.device)).detach()

        for client in self.clients:
            start_time = time.time()

            client.set_TGCR(TGCR)

            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

    def receive_classifiers(self):
        assert len(self.selected_clients) > 0

        self.uploaded_classifiers = []
        for client in self.selected_clients:
            classifier = client.model.head.weight
            for class_id in range(self.num_classes):
                weight_sample = classifier[class_id].detach().clone()
                self.uploaded_classifiers.append((weight_sample, class_id))

    def update_TGCR(self):
        TGCR_opt = torch.optim.SGD(self.TGCR.parameters(), lr=self.server_learning_rate)
        self.TGCR.train()
        classifier_loader = DataLoader(
            self.uploaded_classifiers, self.batch_size, drop_last=False, shuffle=True
        )
        for e in range(self.server_epochs):
            for classifier_param, y in classifier_loader:
                classifier_param = classifier_param.to(self.device)
                y = torch.tensor(y, dtype=torch.long, device=self.device)
                # y = torch.Tensor(y).type(torch.int64).to(self.device)

                tgcr = self.TGCR(torch.arange(self.num_classes, device=self.device))

                inner_product = torch.matmul(classifier_param, tgcr.T)
                loss = self.CEloss(inner_product, y)

                # param_square = torch.sum(
                #     torch.pow(classifier_param, 2), 1, keepdim=True
                # )
                # tgcr_square = torch.sum(torch.pow(tgcr, 2), 1, keepdim=True)
                # param_into_tgcr = torch.matmul(classifier_param, tgcr.T)
                # dist = param_square - 2 * param_into_tgcr + tgcr_square.T
                # dist = torch.sqrt(dist)
                # loss = self.CEloss(-dist, y)

                TGCR_opt.zero_grad()
                loss.backward()
                TGCR_opt.step()

        print(f"Server loss: {loss.item()}")
        self.uploaded_classifiers = []

        self.TGCR.eval()


class Trainable_Global_Classifier_Regularizer(nn.Module):
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()

        self.device = device

        self.embedings = nn.Embedding(num_classes, feature_dim)
        layers = [nn.Sequential(nn.Linear(feature_dim, server_hidden_dim), nn.ReLU())]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        class_id = torch.tensor(class_id, device=self.device)

        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)

        return out
