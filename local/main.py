from experiment import malicious_rate_exp, dif_num_exp, collusion_exp, eigenvector_centrality_exp
import numpy as np

if __name__ == '__main__':

    # parameter initialization
    SIM_NUM = 1000                # number of simulation
    TRUST_UPDATE = 1             # steps for trust updating
    PEER_NUM = 1000                 # number of peers
    PRE_TRUSTED_RATE = 0.05         # rate of pre-trusted peers
    MIN_CAT_PEER_RATE = 0.1
    CATS_NUM = 10                   # number of peer group
    rates = np.arange(0.1, 0.91, 0.15)    # malicious rates
    nums = np.arange(500, 3501, 500)   # number of peers
    RATE = 0.5

    res_dict_1 = dif_num_exp(nums, sim_num=SIM_NUM, rate=RATE, min_cat_peer_rate=MIN_CAT_PEER_RATE,
                                    cats_num=CATS_NUM, trust_upd=TRUST_UPDATE, pre_trusted_rate=PRE_TRUSTED_RATE)

    res_dict_2 = malicious_rate_exp(rates, sim_num=SIM_NUM, peer_num=PEER_NUM, min_cat_peer_rate=MIN_CAT_PEER_RATE,
                                    cats_num=CATS_NUM, trust_upd=TRUST_UPDATE
                                    , pre_trusted_rate=PRE_TRUSTED_RATE
                                    )

    res_dict_3 = collusion_exp(rates, sim_num=SIM_NUM, peer_num=PEER_NUM, min_cat_peer_rate=MIN_CAT_PEER_RATE,
                                    cats_num=CATS_NUM, trust_upd=TRUST_UPDATE
                                    , pre_trusted_rate=PRE_TRUSTED_RATE
                                    )

    res_dict_4 = eigenvector_centrality_exp(rates, sim_num=SIM_NUM, peer_num=PEER_NUM, min_cat_peer_rate=MIN_CAT_PEER_RATE,
                                    cats_num=CATS_NUM, trust_upd=TRUST_UPDATE
                                    , pre_trusted_rate=PRE_TRUSTED_RATE
                                    )

