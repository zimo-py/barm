import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


f5 = [0.012639999999999998, 0.02432, 0.03812, 0.05164, 0.06639999999999999, 0.07715999999999999, 0.08943999999999999]
f10 = [0.012259999999999998, 0.0248, 0.03646, 0.05, 0.060939999999999994, 0.07246000000000001, 0.08292000000000001]
f15 = [0.012920000000000001, 0.02532, 0.038239999999999996, 0.0524, 0.06484, 0.0754, 0.09]


x = [10, 20, 30, 40, 50, 60, 70]
x_len = np.arange(len(x))
total_width, n = 0.9, 3
width = 0.2
xticks = x_len - (total_width - width) / 2
print(xticks)
plt.figure(figsize=(15, 12), dpi=700)

ax = plt.axes()
plt.grid(axis="y", c='#d2c9eb', linestyle='--', zorder=0)
plt.bar(xticks, f5, width=0.9 * width, label="|P|=5", color="#000000", edgecolor='black', linewidth=2,
        zorder=0)
plt.bar(xticks + width, f10, width=0.9 * width, label="|P|=10", color="#4169e1", edgecolor='black', linewidth=2,
        zorder=0)
plt.bar(xticks + 2*width, f15, width=0.9 * width, label="|P|=15", color="#ffa07a", edgecolor='black', linewidth=2,
        zorder=0)

plt.legend(prop={'family': 'Times New Roman', 'size': 25}, loc='upper left', ncol=1)
# x_len = [-0.1, 0.9, 1.9, 2.9, 3.9, 4.9, 5.9]
# x_len = np.array(x_len)
# print(x_len)
plt.xticks(x_len, x, fontproperties='Times New Roman', fontsize=30)
plt.yticks(fontproperties='Times New Roman', fontsize=30)
plt.ylim(0, 0.2)
plt.xlabel("% of malicious peers", fontproperties='Times New Roman', fontsize=30)
plt.ylabel("Rate of inauthentic downloads", fontproperties='Times New Roman', fontsize=30)

plt.savefig('dif_cat_num.pdf', bbox_inches='tight')
plt.show()