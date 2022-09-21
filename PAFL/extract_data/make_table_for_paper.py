"""
boot_avgの表からpaflが優れている個数をカウントした表を作る
ついでにグラフも作る
"""
import os
import sys 
import argparse
import numpy as np
import math
# 異なる階層のutils, model_utilsをインポートするために必要
sys.path.append("../../")
import numpy as np
import pandas as pd
from collections import defaultdict
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle

if __name__ == "__main__":
  B = 10
  num_k = 5
  num_metrics = 6
  num_mode = 2
  num_theta = 2
  # 定数 metrics名の配列, 列名もこの順番になる
  metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'auc']
  datasets = ['tomita_3', 'tomita_4', 'tomita_7', DataSet.BP, DataSet.MR, DataSet.IMDB, DataSet.MNIST, DataSet.TOXIC]
  method_names = ['ochiai', 'tarantula', 'dstar', 'ochiai2', 'ample']

  # コマンドライン引数から受け取り
  parser = argparse.ArgumentParser()
  parser.add_argument("model_type", type=str, help='type of models')
  parser.add_argument("mode", type=str, choices=['stat', 'non-stat'])
  args = parser.parse_args()
  model_type, mode = args.model_type, args.mode

  if mode == 'non-stat':
    # 統計的検定関係ない方
    for method in method_names:
      print(f'---------- method = {method} ----------')
      # 各bootについてとりあえず平均する
      for n in range(1, 6):
        for theta in [10, 50]:
          avg_cnt = np.zeros((len(datasets), num_metrics), dtype=int)
          load_path = get_path(f"data/extracted_data/{model_type}/pred_table_for_look/{method}/boot_avg/n={n}_theta={theta}.csv")
          save_path = get_path(f"data/extracted_data/{model_type}/pred_table_for_look/{method}/avg_cnt/n={n}_theta={theta}.csv")
          boot_mean_csv = pd.read_csv(load_path, header=None)
          for ds_idx, i in enumerate(range(0, len(datasets)*num_k, num_k)):
            boot_mean = boot_mean_csv[i:i+num_k].reset_index()
            for met_idx, j in enumerate(range(1, 2*num_metrics+1, 2)):
              rand = float(boot_mean[j][0])
              avg_cnt[ds_idx][met_idx] = (boot_mean[j-1] < rand).sum()
          # print(avg_cnt)
          os.makedirs(os.path.dirname(save_path), exist_ok=True)
          np.savetxt(save_path, avg_cnt, delimiter=',', fmt='%d')
          print(f'saved in {save_path}')

  elif mode == 'stat':
    # 統計的検定の方
    for method in method_names:
      print(f'---------- method = {method} ----------')
      # 各bootについてとりあえず平均する
      for n in range(1, 6):
        for theta in [10, 50]:
          avg_cnt = np.zeros((len(datasets), num_metrics), dtype=int)
          load_path = get_path(f"data/extracted_data/{model_type}/stat_result/{method}/boot_avg/n={n}_theta={theta}.csv")
          save_path = get_path(f"data/extracted_data/{model_type}/stat_result/{method}/avg_cnt/n={n}_theta={theta}.csv")
          boot_mean_csv = pd.read_csv(load_path, header=None)
          for ds_idx, i in enumerate(range(0, len(datasets)*num_k, num_k)):
            boot_mean = boot_mean_csv[i:i+num_k].reset_index()
            for met_idx in range(num_metrics):
              cnt = 0
              for val in boot_mean[met_idx]:
                if isinstance(val, float):
                  continue
                # 統計的に有意かどうか
                if val.endswith('**'):
                  # 効果量の値を取り出す
                  eff = float(val.rstrip('*'))
                  if eff >= 0.474:
                    cnt += 1
              avg_cnt[ds_idx][met_idx] = cnt
          # print(avg_cnt)
          os.makedirs(os.path.dirname(save_path), exist_ok=True)
          np.savetxt(save_path, avg_cnt, delimiter=',', fmt='%d')
          print(f'saved in {save_path}')