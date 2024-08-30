import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

#增加节点数
# step = [500, 1000, 1500, 2000, 2500, 3000, 3500]
# c3 = [0.057679999999999995, 0.10787999999999998, 0.12654, 0.13198000000000001, 0.14838, 0.15606, 0.165114]
# c4 = [0.034082, 0.034440000000000005, 0.034319999999999996, 0.03606, 0.03964, 0.0403999999999999996, 0.04246]
# # c5 = [0.024546, 0.02366, 0.02066, 0.0201, 0.0219020000000000002, 0.02246, 0.023]
# c6 = [0.021980000000000003, 0.01768, 0.01788, 0.0182, 0.016479999999999998, 0.019340000000000003, 0.01714]

#增加异常率
# step = [10, 20, 30, 40, 50, 60, 70]
# c3 = [0.10128, 0.19867999999999997, 0.30156, 0.3958, 0.5039999999999999, 0.59944, 0.6989999999999998]
# c4 = [0.018080000000000002, 0.04176, 0.14192, 0.26020000000000002, 0.43424, 0.65396, 0.7939600000000001]
# c5 = [0.014960000000000001, 0.033240000000000006, 0.06024, 0.17271999999999998, 0.34492000000000003, 0.57224, 0.7279199999999998]
# c6 = [0.01188, 0.026439999999999998, 0.03764, 0.05267999999999999, 0.06323999999999999, 0.07648000000000002, 0.08924]

# 增加异常行为指数
# step = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# c3 = [0.0854, 0.15408, 0.24732000000000004, 0.3312, 0.43182, 0.5294200000000001, 0.62868, 0.7342799999999999, 0.84504]
# c4 = [0.022099999999999998, 0.04078, 0.0798, 0.14292000000000002, 0.20124, 0.261822, 0.3222399999999999, 0.39400000000001, 0.47444]
# c5 = [0.01494, 0.03314, 0.06572, 0.09498, 0.11696, 0.14587999999999998, 0.20779999999999998, 0.31352, 0.525399999999999]
# c6 = [0.013239999999999998, 0.025240000000000002, 0.0355, 0.04892, 0.060719999999999996, 0.0721, 0.08276, 0.09428, 0.10662]

# 合谋攻击1000节点
# step = [10, 20, 30, 40, 50, 60, 70]
# c3 = [0.10174000000000001, 0.22074, 0.36438, 0.49964, 0.63972000000000005, 0.7806400000000001, 0.90228]
# c4 = [0.01648, 0.04816, 0.1386, 0.2858, 0.53987999999999994, 0.7957200000000001, 0.924000000000001]
# c5 = [0.014120000000000002, 0.03512, 0.06752, 0.1542, 0.39032, 0.71704, 0.8981999999999999]
# c6 = [0.012919999999999997, 0.0261, 0.0384, 0.05818, 0.07535999999999999, 0.0938, 0.1205]
# step = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# c3 = [0.810174000000000001, 0.72074, 0.646438, 0.549964, 0.46972000000000005, 0.404806400000000001, 0.346228, 0.2642647, 0.1734, 0.13436]
# c4 = [0.610174000000000001, 0.52074, 0.436438, 0.37964, 0.33972000000000005, 0.304806400000000001, 0.26228, 0.202647, 0.16734, 0.12436]
# # c5 = [0.014120000000000002, 0.03512, 0.06752, 0.1542, 0.39032, 0.71704, 0.8981999999999999]
# c6 = [0.11919999999999997, 0.09261, 0.084, 0.075818, 0.06835999999999999, 0.05938, 0.051205, 0.046, 0.03826, 0.027064]

# 合谋攻击2000节点
# step = [10, 20, 30, 40, 50, 60, 70]
# c3 = [0.10174000000000001, 0.22074, 0.36438, 0.49964, 0.63972000000000005, 0.7806400000000001, 0.90228]
# c4 = [0.01424, 0.04202, 0.12182, 0.26546000000000003, 0.51772, 0.79308, 0.9191]
# c5 = [0.01488, 0.02998, 0.0613, 0.12840000000000001, 0.37124, 0.6939599999999999, 0.89172]
# c6 = [0.012919999999999997, 0.0251, 0.0374, 0.05718, 0.07235999999999999, 0.0918, 0.11438]

# 合谋攻击3000节点
# step = [10, 20, 30, 40, 50, 60, 70]
# c3 = [0.10174000000000001, 0.22074, 0.36438, 0.49964, 0.63972000000000005, 0.7806400000000001, 0.90228]
# c4 = [0.0157, 0.044840000000000005, 0.11295999999999999, 0.25274, 0.4654, 0.75742, 0.89528]
# c5 = [0.01336, 0.02272, 0.060719999999999996, 0.12752, 0.35214, 0.6753600000000001, 0.876999999999998]
# c6 = [0.0118, 0.0249, 0.0366, 0.05222, 0.064780000000000015, 0.08238, 0.098456]

# 中心向量攻击
# step = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# c3 = [0.7810174000000000001, 0.7074, 0.626438, 0.539964, 0.45972000000000005, 0.394806400000000001, 0.326228, 0.242647, 0.1634, 0.12436]
# c4 = [0.8810174000000000001, 0.8352074, 0.7836438, 0.6964, 0.5972000000000005, 0.4806400000000001, 0.36228, 0.262647, 0.16734, 0.10436]
# # c5 = [0.014120000000000002, 0.03512, 0.06752, 0.1542, 0.39032, 0.71704, 0.8981999999999999]
# c6 = [0.11919999999999997, 0.149261, 0.184, 0.2175818, 0.16835999999999999, 0.1338, 0.0951205, 0.0646, 0.03826, 0.027064]


step = [5, 10, 15, 20]
c3 = [0.08374000000000001, 0.08091999999999999, 0.08392000000000001, 0.0854]
c4 = [0.01318, 0.0252, 0.03734, 0.04806]
c5 = [0.01252, 0.025339999999999994, 0.037660000000000006, 0.04814]
c6 = [0.01204, 0.025279999999999997, 0.03568, 0.049019999999999994]

# 用于生成更多数据点，使曲线更加平滑
x_new = np.linspace(np.array(step).min(), np.array(step).max(), 300)
spl1 = make_interp_spline(step, c3, k=3)  # type: BSpline
spl2 = make_interp_spline(step, c4, k=3)  # type: BSpline
spl3 = make_interp_spline(step, c5, k=3)  # type: BSpline
spl4 = make_interp_spline(step, c6, k=3)  # type: BSpline
c3_smooth = spl1(x_new)
c4_smooth = spl2(x_new)
c5_smooth = spl3(x_new)
c6_smooth = spl4(x_new)
# 绘制折线平滑图
plt.figure(figsize=(15, 12), dpi=700)

plt.plot(x_new, c3_smooth, c='#000000')
plt.scatter(step, c3, s=100, marker='o', color='#000000', label="Non-trust", linewidths=3)
plt.plot(x_new, c4_smooth, c='#4169e1', )
plt.scatter(step, c4, s=100, marker='v', color='#4169e1', label="EigenTrust", linewidths=3)
plt.plot(x_new, c5_smooth, c='#ff6600')
plt.scatter(step, c5, s=100, marker='*', color='#ff6600', label="HonestPeer", linewidths=3)
plt.plot(x_new, c6_smooth, c='#ff0000')
plt.scatter(step, c6, s=100, marker='+', color='#ff0000', label="AARM", linewidths=3)
plt.tick_params(labelsize=25)

plt.legend(prop={'family': 'Times New Roman', 'size': 20}, loc='upper right', ncol=1)
plt.ylim(0, 0.125)
plt.xlabel('Number of groups', fontproperties='Times New Roman', fontsize=30)
# plt.xlabel('% of malicious spies', fontproperties='Times New Roman', fontsize=30)
plt.ylabel('Top-k % of eigenvector central peers', fontproperties='Times New Roman', fontsize=30)
plt.savefig('top-k.pdf', bbox_inches='tight')
plt.show()
