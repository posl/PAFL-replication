import os
import sys
import csv
from collections import defaultdict
# 異なる階層のutils, model_utilsをインポートするために必要
sys.path.append("../../")
import numpy as np
import pandas as pd
from scipy import stats
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.stat_util import *
from utils.help_func import load_pickle

if __name__ == "__main__":
  os.makedirs('./stats_results/cnt', exist_ok=True)
  for n in range(1, 6):
    for theta in [10, 50]:
      with open(f'./stats_results/raw/n={n}_theta={theta}.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        arr = np.array(list(reader))
      # print(arr.shape)

      # count
      ret = np.zeros((7, 6), dtype=int)
      for di in range(6):
        for mj in range(6):
          cnt = 0
          # kごとに比較
          for k in range(5):
            if arr[di*5+k][mj].endswith('**'):
              if float(arr[di*5+k][mj].rstrip('**')) >= 0.474:
                cnt += 1
          ret[di][mj] = cnt
      ret[6] = np.sum(ret, axis=0)
      pd.DataFrame(ret).to_csv(f'./stats_results/cnt/n={n}_theta={theta}.csv', header=False, index=False)