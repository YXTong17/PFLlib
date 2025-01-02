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
import random
import time

import torch
from flcore.clients.clientfgac import (
    clientFGAC,
    clientFGAC_MCD,
    clientFGAC_NTD,
    clientFGAC_Frozen,
    clientFGAC_CC,
)
from flcore.servers.serverbase import Server


class FedFGAC(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFGAC)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients),
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        # self.uploaded_models = []
        self.uploaded_updates = []
        self.uploaded_classifier_weights = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = (
                    client.train_time_cost["total_cost"]
                    / client.train_time_cost["num_rounds"]
                    + client.send_time_cost["total_cost"]
                    / client.send_time_cost["num_rounds"]
                )
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                # self.uploaded_models.append(client.model)
                client_model = copy.deepcopy(client.model)
                for client_param, server_param in zip(
                    client_model.parameters(), self.global_model.parameters()
                ):
                    client_param.data -= server_param.data
                self.uploaded_updates.append(client_model)
                self.uploaded_classifier_weights.append(client.label_counts)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert len(self.uploaded_updates) > 0

        # self.old_global_head = copy.deepcopy(self.global_model.head)
        # self.global_model = copy.deepcopy(self.uploaded_models[0])
        # for param in self.global_model.parameters():
        #     param.data.zero_()

        client_classifier_weights = torch.stack(self.uploaded_classifier_weights)
        total_classifier_weights = client_classifier_weights.sum(dim=0)
        classifier_weights = client_classifier_weights / total_classifier_weights.clamp(
            min=1e-6
        )
        # for i in client_classifier_weights:
        #     print(i)

        for w, client_update, classifier_weight in zip(
            self.uploaded_weights, self.uploaded_updates, classifier_weights
        ):
            # 聚合除最后一层全连接层之外的所有层
            for server_param, client_param_update in zip(
                self.global_model.base.parameters(), client_update.base.parameters()
            ):
                server_param.data += client_param_update.data.clone() * w
            # 聚合最后一层全连接层
            for server_param, client_param_update in zip(
                self.global_model.head.parameters(), client_update.head.parameters()
            ):
                for class_idx, c_w in enumerate(classifier_weight):
                    server_param.data[class_idx] += (
                        client_param_update.data[class_idx] * c_w
                    )
        # for server_param, old_param in zip(
        #     self.global_model.head.parameters(), self.old_global_head.parameters()
        # ):
        #     for cls_idx, total_w in enumerate(total_classifier_weights):
        #         if total_w == 0:
        #             server_param.data[cls_idx] = old_param.data[cls_idx]

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
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFGAC)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


class FedFGAC_NTD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFGAC_NTD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients),
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        # self.uploaded_models = []
        self.uploaded_updates = []
        self.uploaded_classifier_weights = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = (
                    client.train_time_cost["total_cost"]
                    / client.train_time_cost["num_rounds"]
                    + client.send_time_cost["total_cost"]
                    / client.send_time_cost["num_rounds"]
                )
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                # self.uploaded_models.append(client.model)
                client_model = copy.deepcopy(client.model)
                for client_param, server_param in zip(
                    client_model.parameters(), self.global_model.parameters()
                ):
                    client_param.data -= server_param.data
                self.uploaded_updates.append(client_model)
                self.uploaded_classifier_weights.append(client.label_counts)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert len(self.uploaded_updates) > 0

        # self.old_global_head = copy.deepcopy(self.global_model.head)
        # self.global_model = copy.deepcopy(self.uploaded_models[0])
        # for param in self.global_model.parameters():
        #     param.data.zero_()

        client_classifier_weights = torch.stack(self.uploaded_classifier_weights)
        total_classifier_weights = client_classifier_weights.sum(dim=0)
        classifier_weights = client_classifier_weights / total_classifier_weights.clamp(
            min=1e-6
        )

        for w, client_update, classifier_weight in zip(
            self.uploaded_weights, self.uploaded_updates, classifier_weights
        ):
            # 聚合除最后一层全连接层之外的所有层
            for server_param, client_param_update in zip(
                self.global_model.base.parameters(), client_update.base.parameters()
            ):
                server_param.data += client_param_update.data.clone() * w
            # 聚合最后一层全连接层
            for server_param, client_param_update in zip(
                self.global_model.head.parameters(), client_update.head.parameters()
            ):
                for class_idx, c_w in enumerate(classifier_weight):
                    server_param.data[class_idx] += (
                        client_param_update.data[class_idx] * c_w
                    )
        # for server_param, old_param in zip(
        #     self.global_model.head.parameters(), self.old_global_head.parameters()
        # ):
        #     for cls_idx, total_w in enumerate(total_classifier_weights):
        #         if total_w == 0:
        #             server_param.data[cls_idx] = old_param.data[cls_idx]

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
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFGAC_NTD)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


class FedFGAC_MCD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFGAC_MCD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients),
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        # self.uploaded_models = []
        self.uploaded_updates = []
        self.uploaded_classifier_weights = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = (
                    client.train_time_cost["total_cost"]
                    / client.train_time_cost["num_rounds"]
                    + client.send_time_cost["total_cost"]
                    / client.send_time_cost["num_rounds"]
                )
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                # self.uploaded_models.append(client.model)
                client_model = copy.deepcopy(client.model)
                for client_param, server_param in zip(
                    client_model.parameters(), self.global_model.parameters()
                ):
                    client_param.data -= server_param.data
                self.uploaded_updates.append(client_model)
                self.uploaded_classifier_weights.append(client.label_counts)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert len(self.uploaded_updates) > 0

        # self.old_global_head = copy.deepcopy(self.global_model.head)
        # self.global_model = copy.deepcopy(self.uploaded_models[0])
        # for param in self.global_model.parameters():
        #     param.data.zero_()

        client_classifier_weights = torch.stack(self.uploaded_classifier_weights)
        total_classifier_weights = client_classifier_weights.sum(dim=0)
        classifier_weights = client_classifier_weights / total_classifier_weights.clamp(
            min=1e-6
        )

        for w, client_update, classifier_weight in zip(
            self.uploaded_weights, self.uploaded_updates, classifier_weights
        ):
            # 聚合除最后一层全连接层之外的所有层
            for server_param, client_param_update in zip(
                self.global_model.base.parameters(), client_update.base.parameters()
            ):
                server_param.data += client_param_update.data.clone() * w
            # 聚合最后一层全连接层
            for server_param, client_param_update in zip(
                self.global_model.head.parameters(), client_update.head.parameters()
            ):
                for class_idx, c_w in enumerate(classifier_weight):
                    server_param.data[class_idx] += (
                        client_param_update.data[class_idx] * c_w
                    )
        # for server_param, old_param in zip(
        #     self.global_model.head.parameters(), self.old_global_head.parameters()
        # ):
        #     for cls_idx, total_w in enumerate(total_classifier_weights):
        #         if total_w == 0:
        #             server_param.data[cls_idx] = old_param.data[cls_idx]

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
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFGAC_MCD)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


class FedFGAC_Frozen(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFGAC_Frozen)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients),
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        # self.uploaded_models = []
        self.uploaded_updates = []
        self.uploaded_classifier_weights = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = (
                    client.train_time_cost["total_cost"]
                    / client.train_time_cost["num_rounds"]
                    + client.send_time_cost["total_cost"]
                    / client.send_time_cost["num_rounds"]
                )
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                # self.uploaded_models.append(client.model)
                client_model = copy.deepcopy(client.model)
                for client_param, server_param in zip(
                    client_model.parameters(), self.global_model.parameters()
                ):
                    client_param.data -= server_param.data
                self.uploaded_updates.append(client_model)
                self.uploaded_classifier_weights.append(client.label_counts)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert len(self.uploaded_updates) > 0

        # self.old_global_head = copy.deepcopy(self.global_model.head)
        # self.global_model = copy.deepcopy(self.uploaded_models[0])
        # for param in self.global_model.parameters():
        #     param.data.zero_()

        client_classifier_weights = torch.stack(self.uploaded_classifier_weights)
        total_classifier_weights = client_classifier_weights.sum(dim=0)
        classifier_weights = client_classifier_weights / total_classifier_weights.clamp(
            min=1e-6
        )
        # for i in client_classifier_weights:
        #     print(i)

        for w, client_update, classifier_weight in zip(
            self.uploaded_weights, self.uploaded_updates, classifier_weights
        ):
            # 聚合除最后一层全连接层之外的所有层
            for server_param, client_param_update in zip(
                self.global_model.base.parameters(), client_update.base.parameters()
            ):
                server_param.data += client_param_update.data.clone() * w
            # 聚合最后一层全连接层
            for server_param, client_param_update in zip(
                self.global_model.head.parameters(), client_update.head.parameters()
            ):
                for class_idx, c_w in enumerate(classifier_weight):
                    server_param.data[class_idx] += (
                        client_param_update.data[class_idx] * c_w
                    )
        # for server_param, old_param in zip(
        #     self.global_model.head.parameters(), self.old_global_head.parameters()
        # ):
        #     for cls_idx, total_w in enumerate(total_classifier_weights):
        #         if total_w == 0:
        #             server_param.data[cls_idx] = old_param.data[cls_idx]

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
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFGAC_Frozen)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


class FedFGAC_CC(FedFGAC):
    def __init__(self, args, times):
        Server.__init__(self, args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFGAC_CC)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
