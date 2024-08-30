import csv

import numpy as np
import random
import math
from tqdm import tqdm_notebook
import skfuzzy as fuzz


class Peer:

    def __init__(self, id, type='honest', malicious_behavior_rate=0., collective=False):
        self.id = id
        self.malicious_behavior_rate = malicious_behavior_rate
        self.type = type
        self.cats = set()
        self.collective = collective

    def download(self, reciever):
        mark = success = reciever.provide()
        fake = False
        if self.type == 'malicious':
            if self.collective and reciever.type == 'malicious':
                mark = True
            else:
                mark = not success
            fake = True
        return {'mark': mark, 'success': success, 'fake': fake}


    def download_collusion2(self, reciever):
        mark = False
        success = True
        fake = True
        return {'mark': mark, 'success': success, 'fake': fake}


    def download_collusion1(self, reciever):
        mark = success = reciever.provide_collusion1()
        fake = False
        return {'mark': mark, 'success': success, 'fake': fake}


    def provide(self):
        if self.type == 'malicious':
            return False
        return True

    def provide_collusion1(self):
        return True

    def add_cat(self, cat):
        self.cats.add(cat)

    def add_cats(self, cats):
        self.cats.update(cats)


class SimpleEnv:

    def __init__(self, num_peers, malicious_rate, min_cat_peer_rate, num_cats,
                 malicious_behavior_rate=0.05,
                 min_cat_peer_num=3,
                 collective=False):
        self.min_cat_peer_num = min_cat_peer_num
        self.min_cat_peer_rate = min_cat_peer_rate
        self.num_cats = num_cats
        self.cats = []
        self.malicious_behavior_rate = malicious_behavior_rate
        self.malicious_rate = malicious_rate
        self.num_peers = num_peers
        self.interactions = []
        self.peers = []
        self.malicious_peers = []
        self.honest_peers = []
        self.convergence = []
        self.simple_marks = np.zeros(self.num_peers)
        self.collective = collective

        malicious_num = math.ceil(self.num_peers * self.malicious_rate)
        for i in range(malicious_num):
            self.peers.append(Peer(id=i, type='malicious', malicious_behavior_rate=malicious_behavior_rate, collective=collective))
            self.malicious_peers.append(Peer(id=i, type='malicious', malicious_behavior_rate=malicious_behavior_rate, collective=collective))
        for i in range(self.num_peers - malicious_num):
            self.peers.append(Peer(id=i + malicious_num, type='honest'))
            self.honest_peers.append(Peer(id=i + malicious_num, type='honest'))

        for cat in range(self.num_cats):
            cat_len = math.ceil(min_cat_peer_rate * self.num_peers)
            peer_subset = set(np.random.choice(self.peers, cat_len, replace=False))
            for p in peer_subset:
                p.add_cat(cat)
            self.cats.append(peer_subset)

        all_cats = set(range(self.num_cats))
        for peer in self.peers:
            if len(peer.cats) < self.min_cat_peer_num:
                cats = set(np.random.choice(list(all_cats - peer.cats), size=self.min_cat_peer_num - len(peer.cats),
                                            replace=False))
                peer.add_cats(cats)
                for cat in cats:
                    self.cats[cat].add(peer)

    def simulate(self, n_inter: int):
        for i in tqdm_notebook(range(n_inter)):
            sender, reciever = self.choose_peers()

            interaction = interact(sender, reciever)
            c = 1 if interaction['mark'] else 0
            self.simple_marks[reciever.id] += c
            self.interactions.append(interaction)

    def choose_peers(self):
        sender = np.random.choice(self.peers)
        cat = np.random.choice(list(sender.cats))
        reciever = np.random.choice(list(self.cats[cat] - {sender}))
        return sender, reciever


class EigenTrustEnv(SimpleEnv):

    def __init__(self, num_peers, malicious_rate, pre_trusted_rate,
                 min_cat_peer_rate,
                 num_cats,
                 trust_upd,
                 malicious_behavior_rate=1.,
                 min_cat_peer_num=3,
                 a=0.1,
                 collective=False):
        super().__init__(num_peers=num_peers,
                         malicious_rate=malicious_rate,
                         malicious_behavior_rate=malicious_behavior_rate,
                         min_cat_peer_rate=min_cat_peer_rate,
                         min_cat_peer_num=min_cat_peer_num,
                         num_cats=num_cats,
                         collective=collective)
        self.trust_upd = trust_upd
        pre_trusted_num = math.ceil(self.num_peers * pre_trusted_rate)  # 计算预信任节点数目
        self.pre_trusted = set()  # 预信任节点初始化
        for peer in self.peers[-pre_trusted_num:]:
            self.pre_trusted.add(peer)
        self.pre_trusted_dist = \
            np.array([1 / len(self.pre_trusted) if peer in self.pre_trusted else 0 for peer in self.peers])  # 预信任节点初始信任值为1/(预信任节点数)
        # 其它节点信任值初始化为0
        self.reputation = np.zeros(self.num_peers)
        self.local_marks = np.zeros((self.num_peers, self.num_peers))
        self.local_trust_matrix = np.zeros((self.num_peers, self.num_peers))
        self.a = a

    def choose_peers(self):
        sender = np.random.choice(self.peers)
        # print("sender:", sender)
        # print("sender.cats:", sender.cats)
        cat = np.random.choice(list(sender.cats))
        # reciever = np.random.choice(self.peers)
        recievers, reps = [s for s in (self.cats[cat] - {sender})], np.array([self.reputation[s.id] for s in
                                                                           (self.cats[cat] - {sender})])
        # print("reps:", reps)
        with_zero = [s for s, r in zip(recievers, reps) if r == 0]

        # reciever = sorted(recievers, key=lambda x: self.reputation[x.id], reverse=True)[0] #deterministic
        s = np.sum(reps)
        if len(with_zero) > 0:
            if np.random.rand() < 0.1 or s == 0:
                reciever = np.random.choice(with_zero)
                return sender, reciever
        probs = [p / s for p in reps]
        reciever = np.random.choice(recievers, p=probs)

        return sender, reciever

    def update_reputation(self):
        eps = 0.001  # 误差阈值
        t = self.pre_trusted_dist.copy()
        t_next = self.pre_trusted_dist.copy()
        dif = eps
        c = 0  # 收敛轮数
        while dif >= eps:
            t_next = (1 - self.a) * self.local_trust_matrix.T.dot(t) + self.a * self.pre_trusted_dist  # 信任值计算公式：(1-a)*C^T + a*p
            d = t_next - t
            dif = np.linalg.norm(d)  # 求d的范数
            t = t_next
            c += 1
        self.convergence.append(c)
        self.reputation = t_next

    def update_local_trust_matrix(self, peer_i, peer_j, mark):
        self.local_marks[peer_i, peer_j] += mark  # 节点i和j的历史交易记录
        s = np.sum([x for x in self.local_marks[peer_i] if x > 0])  # 节点i的历史交易记录
        for j in range(self.num_peers):
            self.local_trust_matrix[peer_i, j] = max(self.local_marks[peer_i, j], 0) / s \
                if s != 0 \
                else self.pre_trusted_dist[peer_j]

    def simulate(self, n_inter: int):
        np.random.seed(42)
        random.seed(42)
        for i in tqdm_notebook(range(n_inter)):
            sender, reciever = self.choose_peers()
            # print(sender.id, reciever.id)
            interaction = interact(sender, reciever)
            mark = 1 if interaction['mark'] else -1
            self.update_local_trust_matrix(sender.id, reciever.id, mark)
            self.interactions.append(interaction)
            if (i + 1) % self.trust_upd == 0:
                self.update_reputation()
            # print(i, self.reputation)
        save_rep('rep_eg.csv', self.reputation)


class BarmEnv(SimpleEnv):

    def __init__(self, num_peers, malicious_rate, pre_trusted_rate,
                 min_cat_peer_rate,
                 num_cats,
                 trust_upd,
                 malicious_behavior_rate=1,
                 min_cat_peer_num=1,
                 a=0.7,
                 collective=False):
        super().__init__(num_peers=num_peers,
                         malicious_rate=malicious_rate,
                         malicious_behavior_rate=malicious_behavior_rate,
                         min_cat_peer_rate=min_cat_peer_rate,
                         min_cat_peer_num=min_cat_peer_num,
                         num_cats=num_cats,
                         collective=collective)
        self.trust_upd = trust_upd
        pre_trusted_num = math.ceil(self.num_peers * pre_trusted_rate)  # 计算预信任节点数目
        self.pre_trusted = set()  # 预信任节点初始化
        for peer in self.peers[-pre_trusted_num:]:
            self.pre_trusted.add(peer)
        self.pre_trusted_dist = \
            np.array([1 / len(self.pre_trusted) if peer in self.pre_trusted else 0 for peer in self.peers])  # 预信任节点初始信任值为1/(预信任节点数)
        # 其它节点信任值初始化为0
        self.reputation = self.pre_trusted_dist
        # self.reputation = np.zeros(self.num_peers)
        self.local_marks = np.zeros((self.num_peers, self.num_peers))
        self.local_trust_matrix = np.zeros((self.num_peers, self.num_peers))
        self.a = a

    def choose_peers(self, group):
        # Reputation Average (RA) strategy
        rows = self.num_cats
        cols = math.ceil(self.num_peers / self.num_cats)
        group_avg_rep = []
        for i in range(rows):
            avg_rep = 0
            for j in range(cols):
                avg_rep = avg_rep + self.reputation[group[i][j].id]
            group_avg_rep.append(avg_rep / cols)
        # print(group_avg_rep)

        # Two-Phase Surfer (TPS) strategy
        sender = np.random.choice(self.peers)  # Randomly initialize the task initiator
        position_sender = math.floor(sender.id / (self.num_peers / self.num_cats))  # Locating the sender's group
        print('sender and its group:', sender.id, position_sender)

        # P1-S for the selection of target group
        # print(group_avg_rep)
        del group_avg_rep[position_sender]# Take the complementary set and exclude the group in which the sender is located
        sum_rep = np.sum(group_avg_rep)
        weights = [p / sum_rep for p in group_avg_rep]
        print(weights)
        group_list = [x for x in range(0, self.num_cats)]
        del group_list[position_sender]
        target_group = np.random.choice(group_list, p=weights)  # controls the probability of group selection based on reputation weights
        print('target_group:', target_group)

        # P2-S for the selection of target peer
        candidate_peers = []
        for i in range(cols):
            candidate_peers.append(group[target_group][i])
        sorted_peers = sorted(candidate_peers, key=lambda x: self.reputation[x.id], reverse=True)[
                       :10]
        receiver = np.random.choice(
            sorted_peers)  # Randomly select the object of the transaction from the target group
        print('receiver:', receiver.id)
        sender = sender
        reciever = receiver
        return sender, reciever

    def update_reputation(self, i):
        eps = 0.0001
        t = self.reputation.copy()
        t_next = t + self.local_trust_matrix.T.dot(t) * eps  # Formulation of trust value after removing the influence of pre-trusted nodes：C^T
        print('t', t)
        print('t_next', t_next)
        # t_next = self.reputation.copy()
        # eye = np.eye(self.num_peers)
        # print(t)
        # print(self.local_trust_matrix)
        # dif = eps
        # c = 0  # 收敛轮数
        # while dif >= eps:
        #     # t_next = (1 - self.a) * self.local_trust_matrix.T.dot(t) + self.a * self.pre_trusted_dist  # 信任值计算公式：(1-a)*C^T + a*p
        #     t_next = eye.T.dot(t) + self.local_trust_matrix.T.dot(t)  # Formulation of trust value after removing the influence of pre-trusted nodes：C^T
        #     print('t_next', t_next)
        #     d = t_next - t
        #     dif = np.linalg.norm(d)  # 求d的范数
        #     print(dif)
        #     t = t_next
        #     c += 1
        # self.convergence.append(c)
        self.reputation = t_next

    def update_local_trust_matrix(self, peer_i, peer_j, mark):
        self.local_marks[peer_i, peer_j] += mark  # 节点i和j的历史交易记录
        # print([peer_i, peer_j],self.local_marks[peer_i, peer_j])
        s = np.sum([x for x in self.local_marks[peer_i] if x > 0])  # 节点i的历史交易记录
        # print(s)
        for j in range(self.num_peers):
            self.local_trust_matrix[peer_i, j] = max(self.local_marks[peer_i, j], 0) / s \
                if s != 0 \
                else self.reputation[peer_j]

    def simulate(self, n_inter: int):
        np.random.seed(56)
        random.seed(56)
        sorted_rep = []
        sorted_peers = sorted(self.peers, key=lambda x: self.reputation[x.id], reverse=True)[
                       :self.num_peers]  # k = num_cats
        for peer in sorted_peers:
            sorted_rep.append(self.reputation[peer.id])
        # print(sorted_rep)
        rows = self.num_cats
        cols = math.ceil(self.num_peers / self.num_cats)
        sorted_arr = [[0 for _ in range(cols)] for _ in range(rows)]
        k = 0
        for i in range(cols):
            for j in range(rows):
                sorted_arr[j][i] = sorted_peers[k]
                k = k + 1
        group = [
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols
        ]
        group_rep = [
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols
        ]
        candidates = list(range(0, 10))
        for i in range(cols):
            m = np.random.choice(candidates, size=10, replace=False)  # Avoiding duplication
            for j in range(rows):
                group[j][i] = sorted_arr[m[j]][i]
                group_rep[j][i] = self.reputation[sorted_arr[m[j]][i].id]
        # print(group)

        for i in tqdm_notebook(range(n_inter)):
            sender, reciever = self.choose_peers(group)
            # print(sender.id, reciever.id)
            interaction = interact(sender, reciever)
            print(interaction)
            mark = 1 if interaction['mark'] else -1
            self.update_local_trust_matrix(sender.id, reciever.id, mark)
            print(self.local_trust_matrix)
            self.interactions.append(interaction)
            if (i + 1) % self.trust_upd == 0:
                self.update_reputation(i)

            transaction = []
            transaction.append(sender.id)
            transaction.append(reciever.id)
            transaction.append(self.reputation[reciever.id])
            # 打开CSV文件进行写入
            with open('transaction.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(transaction)

        print(i, self.reputation)
        save_rep('rep_barm.csv', self.reputation)

class BarmEnvWithCollusion(SimpleEnv):

    def __init__(self, num_peers, malicious_rate, pre_trusted_rate,
                 min_cat_peer_rate,
                 num_cats,
                 trust_upd,
                 malicious_behavior_rate=1,
                 min_cat_peer_num=1,
                 a=0.7,
                 collective=False):
        super().__init__(num_peers=num_peers,
                         malicious_rate=malicious_rate,
                         malicious_behavior_rate=malicious_behavior_rate,
                         min_cat_peer_rate=min_cat_peer_rate,
                         min_cat_peer_num=min_cat_peer_num,
                         num_cats=num_cats,
                         collective=collective)
        self.trust_upd = trust_upd
        pre_trusted_num = math.ceil(self.num_peers * pre_trusted_rate)  # 计算预信任节点数目
        self.pre_trusted = set()  # 预信任节点初始化
        for peer in self.peers[-pre_trusted_num:]:
            self.pre_trusted.add(peer)
        self.pre_trusted_dist = \
            np.array([1 / len(self.pre_trusted) if peer in self.pre_trusted else 0 for peer in self.peers])  # 预信任节点初始信任值为1/(预信任节点数)
        # 其它节点信任值初始化为0
        self.reputation = self.pre_trusted_dist
        # self.reputation = np.zeros(self.num_peers)
        self.local_marks = np.zeros((self.num_peers, self.num_peers))
        self.local_trust_matrix = np.zeros((self.num_peers, self.num_peers))
        self.a = a

    def choose_peers(self, group):
        # Reputation Average (RA) strategy
        rows = self.num_cats
        cols = math.ceil(self.num_peers / self.num_cats)
        group_avg_rep = []
        for i in range(rows):
            avg_rep = 0
            for j in range(cols):
                avg_rep = avg_rep + self.reputation[group[i][j].id]
            group_avg_rep.append(avg_rep / cols)
        # print(group_avg_rep)

        # Two-Phase Surfer (TPS) strategy
        sender = np.random.choice(self.peers)  # Randomly initialize the task initiator
        position_sender = math.floor(sender.id / (self.num_peers / self.num_cats))  # Locating the sender's group
        print('sender and its group:', sender.id, position_sender)

        # P1-S for the selection of target group
        # print(group_avg_rep)
        del group_avg_rep[position_sender]# Take the complementary set and exclude the group in which the sender is located
        sum_rep = np.sum(group_avg_rep)
        weights = [p / sum_rep for p in group_avg_rep]
        print(weights)
        group_list = [x for x in range(0, self.num_cats)]
        del group_list[position_sender]
        target_group = np.random.choice(group_list, p=weights)  # controls the probability of group selection based on reputation weights
        print('target_group:', target_group)

        # P2-S for the selection of target peer
        candidate_peers = []
        for i in range(cols):
            candidate_peers.append(group[target_group][i])
        sorted_peers = sorted(candidate_peers, key=lambda x: self.reputation[x.id], reverse=True)[
                       :10]
        reciever = np.random.choice(
            sorted_peers)  # Randomly select the object of the transaction from the target group
        print('reciever:', reciever.id)
        return sender, reciever

    def update_reputation(self, i):
        eps = 0.0001
        t = self.reputation.copy()
        t_next = t + self.local_trust_matrix.T.dot(t) * eps  # Formulation of trust value after removing the influence of pre-trusted nodes：C^T
        print('t', t)
        print('t_next', t_next)
        # t_next = self.reputation.copy()
        # eye = np.eye(self.num_peers)
        # print(t)
        # print(self.local_trust_matrix)
        # dif = eps
        # c = 0  # 收敛轮数
        # while dif >= eps:
        #     # t_next = (1 - self.a) * self.local_trust_matrix.T.dot(t) + self.a * self.pre_trusted_dist  # 信任值计算公式：(1-a)*C^T + a*p
        #     t_next = eye.T.dot(t) + self.local_trust_matrix.T.dot(t)  # Formulation of trust value after removing the influence of pre-trusted nodes：C^T
        #     print('t_next', t_next)
        #     d = t_next - t
        #     dif = np.linalg.norm(d)  # 求d的范数
        #     print(dif)
        #     t = t_next
        #     c += 1
        # self.convergence.append(c)
        self.reputation = t_next

    def update_local_trust_matrix(self, peer_i, peer_j, mark):
        self.local_marks[peer_i, peer_j] += mark  # 节点i和j的历史交易记录
        # print([peer_i, peer_j],self.local_marks[peer_i, peer_j])
        s = np.sum([x for x in self.local_marks[peer_i] if x > 0])  # 节点i的历史交易记录
        # print(s)
        for j in range(self.num_peers):
            self.local_trust_matrix[peer_i, j] = max(self.local_marks[peer_i, j], 0) / s \
                if s != 0 \
                else self.reputation[peer_j]

    def simulate(self, n_inter: int):
        np.random.seed(56)
        random.seed(56)
        sorted_rep = []
        sorted_peers = sorted(self.peers, key=lambda x: self.reputation[x.id], reverse=True)[
                       :self.num_peers]  # k = num_cats
        for peer in sorted_peers:
            sorted_rep.append(self.reputation[peer.id])
        # print(sorted_rep)
        rows = self.num_cats
        cols = math.ceil(self.num_peers / self.num_cats)
        sorted_arr = [[0 for _ in range(cols)] for _ in range(rows)]
        k = 0
        for i in range(cols):
            for j in range(rows):
                sorted_arr[j][i] = sorted_peers[k]
                k = k + 1
        group = [
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols
        ]
        group_rep = [
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols
        ]
        candidates = list(range(0, 10))
        for i in range(cols):
            m = np.random.choice(candidates, size=10, replace=False)  # Avoiding duplication
            for j in range(rows):
                group[j][i] = sorted_arr[m[j]][i]
                group_rep[j][i] = self.reputation[sorted_arr[m[j]][i].id]
        # print(group)

        for i in tqdm_notebook(range(n_inter)):
            sender, reciever = self.choose_peers(group)
            # print(sender.id, reciever.id)
            if sender.type == 'malicious' and reciever.type == 'malicious':
                interaction = interact_collusion1(sender, reciever)  # Malicious nodes harboring each other
            elif sender.type == 'malicious' and reciever.type == 'honest':
                interaction = interact_collusion2(sender, reciever)  # Malicious nodes deface honest nodes
            else:
                interaction = interact(sender, reciever)
            print(interaction)
            mark = 1 if interaction['mark'] else -1
            self.update_local_trust_matrix(sender.id, reciever.id, mark)
            print(self.local_trust_matrix)
            self.interactions.append(interaction)
            if (i + 1) % self.trust_upd == 0:
                self.update_reputation(i)
        print(i, self.reputation)
        save_rep('rep_barm_collusion.csv', self.reputation)

class BarmEnvWithEigenvectorCentralityAttack(SimpleEnv):

    def __init__(self, num_peers, malicious_rate, pre_trusted_rate,
                 min_cat_peer_rate,
                 num_cats,
                 trust_upd,
                 malicious_behavior_rate=0.5,
                 min_cat_peer_num=1,
                 a=0.7,
                 collective=False):
        super().__init__(num_peers=num_peers,
                         malicious_rate=malicious_rate,
                         malicious_behavior_rate=malicious_behavior_rate,
                         min_cat_peer_rate=min_cat_peer_rate,
                         min_cat_peer_num=min_cat_peer_num,
                         num_cats=num_cats,
                         collective=collective)
        self.trust_upd = trust_upd
        pre_trusted_num = math.ceil(self.num_peers * pre_trusted_rate)  # 计算预信任节点数目
        self.pre_trusted = set()  # 预信任节点初始化
        for peer in self.peers[-pre_trusted_num:]:
            self.pre_trusted.add(peer)
        self.pre_trusted_dist = \
            np.array([1 / len(self.pre_trusted) if peer in self.pre_trusted else 0 for peer in self.peers])  # 预信任节点初始信任值为1/(预信任节点数)
        # 其它节点信任值初始化为0
        self.reputation = self.pre_trusted_dist
        # self.reputation = np.zeros(self.num_peers)
        self.local_marks = np.zeros((self.num_peers, self.num_peers))
        self.local_trust_matrix = np.zeros((self.num_peers, self.num_peers))
        self.a = a

    def choose_peers(self, group):
        # Reputation Average (RA) strategy
        rows = self.num_cats
        cols = math.ceil(self.num_peers / self.num_cats)
        group_avg_rep = []
        for i in range(rows):
            avg_rep = 0
            for j in range(cols):
                avg_rep = avg_rep + self.reputation[group[i][j].id]
            group_avg_rep.append(avg_rep / cols)
        # print(group_avg_rep)

        # Two-Phase Surfer (TPS) strategy
        sender = np.random.choice(self.peers)  # Randomly initialize the task initiator
        position_sender = math.floor(sender.id / (self.num_peers / self.num_cats))  # Locating the sender's group
        print('sender and its group:', sender.id, position_sender)
        if sender in self.honest_peers:  # Honest nodes choose trading nodes according to the strategy
            # P1-S for the selection of target group
            # print(group_avg_rep)
            del group_avg_rep[position_sender]# Take the complementary set and exclude the group in which the sender is located
            sum_rep = np.sum(group_avg_rep)
            weights = [p / sum_rep for p in group_avg_rep]
            print(weights)
            group_list = [x for x in range(0, self.num_cats)]
            del group_list[position_sender]
            target_group = np.random.choice(group_list, p=weights)  # controls the probability of group selection based on reputation weights
            print('target_group:', target_group)

            # P2-S for the selection of target peer
            candidate_peers = []
            for i in range(cols):
                candidate_peers.append(group[target_group][i])
            sorted_peers = sorted(candidate_peers, key=lambda x: self.reputation[x.id], reverse=True)[
                           :10]
            reciever = np.random.choice(
                sorted_peers)  # Randomly select the object of the transaction from the target group
            print('reciever:', reciever.id)
        else:  # Malicious node selects honest node with large trust value and intentionally scores it low
            sorted_honest_peers = sorted(self.honest_peers, key=lambda x: self.reputation[x.id], reverse=True)[
                           :10]
            reciever = np.random.choice(sorted_honest_peers)
        return sender, reciever


    def update_reputation(self, i):
        eps = 0.0001
        t = self.reputation.copy()
        t_next = t + self.local_trust_matrix.T.dot(t) * eps  # Formulation of trust value after removing the influence of pre-trusted nodes：C^T
        print('t', t)
        print('t_next', t_next)
        # t_next = self.reputation.copy()
        # eye = np.eye(self.num_peers)
        # print(t)
        # print(self.local_trust_matrix)
        # dif = eps
        # c = 0  # 收敛轮数
        # while dif >= eps:
        #     # t_next = (1 - self.a) * self.local_trust_matrix.T.dot(t) + self.a * self.pre_trusted_dist  # 信任值计算公式：(1-a)*C^T + a*p
        #     t_next = eye.T.dot(t) + self.local_trust_matrix.T.dot(t)  # Formulation of trust value after removing the influence of pre-trusted nodes：C^T
        #     print('t_next', t_next)
        #     d = t_next - t
        #     dif = np.linalg.norm(d)  # 求d的范数
        #     print(dif)
        #     t = t_next
        #     c += 1
        # self.convergence.append(c)
        self.reputation = t_next

    def update_local_trust_matrix(self, peer_i, peer_j, mark):
        self.local_marks[peer_i, peer_j] += mark  # 节点i和j的历史交易记录
        # print([peer_i, peer_j],self.local_marks[peer_i, peer_j])
        s = np.sum([x for x in self.local_marks[peer_i] if x > 0])  # 节点i的历史交易记录
        # print(s)
        for j in range(self.num_peers):
            self.local_trust_matrix[peer_i, j] = max(self.local_marks[peer_i, j], 0) / s \
                if s != 0 \
                else self.reputation[peer_j]

    def simulate(self, n_inter: int):
        np.random.seed(56)
        random.seed(56)
        sorted_rep = []
        sorted_peers = sorted(self.peers, key=lambda x: self.reputation[x.id], reverse=True)[
                       :self.num_peers]  # k = num_cats
        for peer in sorted_peers:
            sorted_rep.append(self.reputation[peer.id])
        # print(sorted_rep)
        rows = self.num_cats
        cols = math.ceil(self.num_peers / self.num_cats)
        sorted_arr = [[0 for _ in range(cols)] for _ in range(rows)]
        k = 0
        for i in range(cols):
            for j in range(rows):
                sorted_arr[j][i] = sorted_peers[k]
                k = k + 1
        group = [
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols
        ]
        group_rep = [
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols,
            [None] * cols
        ]
        candidates = list(range(0, 10))
        for i in range(cols):
            m = np.random.choice(candidates, size=10, replace=False)  # Avoiding duplication
            for j in range(rows):
                group[j][i] = sorted_arr[m[j]][i]
                group_rep[j][i] = self.reputation[sorted_arr[m[j]][i].id]
        # print(group)

        for i in tqdm_notebook(range(n_inter)):
            sender, reciever = self.choose_peers(group)
            # print(sender.id, reciever.id)
            if sender.type == 'malicious' and reciever.type == 'malicious':
                interaction = interact_collusion1(sender, reciever)  # Malicious nodes harboring each other
            elif sender.type == 'malicious' and reciever.type == 'honest':
                interaction = interact_collusion2(sender, reciever)  # Malicious nodes deface honest nodes
            else:
                interaction = interact(sender, reciever)
            print(interaction)
            mark = 1 if interaction['mark'] else -1
            self.update_local_trust_matrix(sender.id, reciever.id, mark)
            print(self.local_trust_matrix)
            self.interactions.append(interaction)
            if (i + 1) % self.trust_upd == 0:
                self.update_reputation(i)
        print(i, self.reputation)
        save_rep('rep_barm_collusion.csv', self.reputation)

def save_rep(filename, reputation):
    for rep in reputation:
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([rep])

def interact(sender: Peer, reciever: Peer):
    info = sender.download(reciever)
    return info

# collusion model 2: Malicious nodes deface honest nodes
def interact_collusion2(sender: Peer, reciever: Peer):
    info = sender.download_collusion2(reciever)
    return info

# collusion model 1: Malicious nodes harboring each other
def interact_collusion1(sender: Peer, reciever: Peer):
    info = sender.download_collusion1(reciever)
    return info

def count_stat(interactions: list):
    return sum([1 for x in interactions if not x['success']]) / len(interactions)
