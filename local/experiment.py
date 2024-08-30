import csv

from environment import *
# from environment2 import *
import matplotlib.pyplot as plt


def malicious_rate_exp(rates, sim_num, peer_num, min_cat_peer_rate, cats_num, trust_upd, pre_trusted_rate, exp_iter=5):
    for rate in rates:
        for i in range(exp_iter):
            simple = SimpleEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                               num_cats=cats_num)
            simple.simulate(sim_num)
            for row in simple.interactions:
                row_list = []
                row_list.append(row['mark'])
                row_list.append(row['success'])
                row_list.append(row['fake'])

                # 打开CSV文件进行写入
                with open('mark_simple.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row_list)

        for i in range(exp_iter):
            eigen = EigenTrustEnv(num_peers=peer_num, malicious_rate=rate, pre_trusted_rate=pre_trusted_rate,
                                  min_cat_peer_rate=min_cat_peer_rate, num_cats=cats_num, trust_upd=trust_upd)
            eigen.simulate(sim_num)
            for row in eigen.interactions:
                row_list = []
                row_list.append(row['mark'])
                row_list.append(row['success'])
                row_list.append(row['fake'])

                # 打开CSV文件进行写入
                with open('mark_eg.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row_list)

        for i in range(exp_iter):
            barm = BarmEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd, pre_trusted_rate=pre_trusted_rate)
            barm.simulate(sim_num)
            for row in barm.interactions:
                row_list = []
                row_list.append(row['mark'])
                row_list.append(row['success'])
                row_list.append(row['fake'])

                # 打开CSV文件进行写入
                with open('mark_barm.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row_list)

        print(f'Rate {rate:.3f} completed!')

def collusion_exp(rates, sim_num, peer_num, min_cat_peer_rate, cats_num, trust_upd, pre_trusted_rate, exp_iter=5):
    for rate in rates:
        for i in range(exp_iter):
            barm = BarmEnvWithCollusion(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd, pre_trusted_rate=pre_trusted_rate)
            barm.simulate(sim_num)
            for row in barm.interactions:
                row_list = []
                row_list.append(row['mark'])
                row_list.append(row['success'])
                row_list.append(row['fake'])

                # 打开CSV文件进行写入
                with open('mark_barm_collusion.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row_list)

        print(f'Rate {rate:.3f} completed!')


def eigenvector_centrality_exp(rates, sim_num, peer_num, min_cat_peer_rate, cats_num, trust_upd, pre_trusted_rate, exp_iter=5):
    for rate in rates:
        for i in range(exp_iter):
            barm = BarmEnvWithEigenvectorCentralityAttack(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd, pre_trusted_rate=pre_trusted_rate)
            barm.simulate(sim_num)
            for row in barm.interactions:
                row_list = []
                row_list.append(row['mark'])
                row_list.append(row['success'])
                row_list.append(row['fake'])

                # 打开CSV文件进行写入
                with open('mark_barm_ECA.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row_list)

        print(f'Rate {rate:.3f} completed!')


def dif_num_exp(nums, sim_num, rate, min_cat_peer_rate, cats_num, trust_upd, pre_trusted_rate, exp_iter=5):
    res_dict = {'nums': nums, 'simple': [], 'eigen': [], 'honest': [], 'peer': [], 'abs': [], 'peer_eigen': [],
                'peer_fuzzy': []}
    for num in nums:
        t = []
        for i in range(exp_iter):
            eigen = EigenTrustEnv(num_peers=num, malicious_rate=rate, pre_trusted_rate=pre_trusted_rate,
                                  min_cat_peer_rate=min_cat_peer_rate, num_cats=cats_num, trust_upd=trust_upd)
            eigen.simulate(sim_num)
            for row in eigen.interactions:
                row_list = []
                row_list.append(row['mark'])
                row_list.append(row['success'])
                row_list.append(row['fake'])

                # 打开CSV文件进行写入
                with open('dif_num_eg.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row_list)

        t = []
        for i in range(exp_iter):
            barm = BarmEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                           num_cats=cats_num, trust_upd=trust_upd, pre_trusted_rate=pre_trusted_rate)
            barm.simulate(sim_num)
            for row in barm.interactions:
                row_list = []
                row_list.append(row['mark'])
                row_list.append(row['success'])
                row_list.append(row['fake'])

                # 打开CSV文件进行写入
                with open('dif_num_barm.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row_list)

        print(f'Num {num} completed!')
    print(res_dict)
