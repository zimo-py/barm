import matplotlib.pyplot as plt


step = [5, 10, 15, 20]
x = range(len(step))
c3 = [0.08374000000000001, 0.08091999999999999, 0.08392000000000001, 0.0854]
c4 = [0.01318, 0.0252, 0.03734, 0.04806]
c5 = [0.01252, 0.025339999999999994, 0.037660000000000006, 0.04814]
c6 = [0.01204, 0.025279999999999997, 0.03568, 0.049019999999999994]


x = range(len(x))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字


c3 = [0.081091999999999999, 0.0836, 0.0853734, 0.10806]
c4 = [0.08374000000000001, 0.08091999999999999, 0.08392000000000001, 0.0854]
c5 = [0.08252, 0.0839339999999999994, 0.08037660000000000006, 0.08614]
c6 = [0.085204, 0.0825279999999999997, 0.083568, 0.08049019999999999994]


plt.figure(figsize=(15, 12), dpi=700)
plt.plot(x, c3, color='blueviolet', marker='o', markersize=10, linestyle='-', label='top_k=5', linewidth=2)
plt.plot(x, c4,  color='red', marker='v', markersize=10, linestyle='-', label='top_k=10', linewidth=2)
plt.plot(x, c5,  color='green', marker='+', markersize=10, linestyle='-', label='top_k=15', linewidth=2)
plt.plot(x, c6,  color='black', marker='*', markersize=10, linestyle='-', label='top_k=20', linewidth=2)
plt.legend(prop={'family': 'Times New Roman', 'size': 30}, loc='upper left', ncol=1)
plt.ylim(0.05, 0.125)
plt.yticks(fontsize=30)
plt.xticks(x, step, fontsize=30)
plt.xlabel('Number of groups', fontproperties='Times New Roman', fontsize=30)
plt.ylabel('Top-k % of eigenvector central peers', fontproperties='Times New Roman', fontsize=30)
plt.savefig('top-k.pdf', bbox_inches='tight')
plt.show()
