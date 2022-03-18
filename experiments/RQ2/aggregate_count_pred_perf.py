import os
import sys 
# 異なる階層のutils, model_utilsをインポートするために必要
sys.path.append("../../")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import csv
from collections import defaultdict
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle

if __name__ == "__main__":  
  os.makedirs('./pred_results', exist_ok=True)
  n_cnt = np.zeros((5, ))
  # aggregate
  for n in range(1, 6):
    for theta in [10, 50]:
      arr = np.zeros((30, 12), dtype=float)
      for i in range(10):
        pred_res_save_path = ExtractData.PRED_RESULT_FOR_LOOK.format(i, n, theta)
        with open(pred_res_save_path, 'r') as csv_file:
          reader = csv.reader(csv_file)
          list_of_rows = np.array(list(reader))
          arr += np.array(list_of_rows[1:, 2:], dtype=float)

      # count
      ret = np.zeros((7, 6), dtype=int)
      for di in range(6):
        for mj in range(6):
          rand = arr[di*5][2*mj+1]
          cnt = 0
          # kごとに比較
          for k in range(5):
            if arr[di*5+k][2*mj] < rand:
              cnt += 1 
          ret[di][mj] = cnt
      ret[6] = np.sum(ret, axis=0)
      n_cnt[n-1] += np.sum(ret[6])
      pd.DataFrame(ret).to_csv(f'./pred_results/n={n}_theta={theta}.csv', header=False, index=False)

  # plot effect of n
  n = np.array([1, 2, 3, 4, 5])
  r = np.round(100 * n_cnt / 360).astype(int)

  plt.figure(figsize=(14, 8))
  plt.rcParams["font.size"] = 28
  plt.ylim([0, 100])
  plt.yticks([0, 60, 80, 100], ['0%', '60%', '80%', '100%\n(360)'])

  plt.bar(n, r, color='white', edgecolor='black', linewidth=2.5)
  plt.xlabel("$n$")
  for x, y, s in zip(n, r, n_cnt):
      plt.text(x, y, f'{y}%\n({s})', ha='center', va='bottom')
  plt.savefig("./rq2_count.pdf", format='pdf', dpi=300, bbox_inches='tight')