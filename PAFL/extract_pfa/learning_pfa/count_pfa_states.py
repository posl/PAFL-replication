"""
PFAの状態数をカウントしてcsvに吐き出す
"""
import argparse
import os
import sys 
import re
import glob
import csv
import numpy as np
#同じ階層のディレクトリからインポートするために必要
sys.path.append("../")
# 異なる階層のmodel_utils, utilsをインポートするために必要
sys.path.append("../../../")
# 異なる階層utilsからインポート
from utils.constant import *

if __name__=='__main__':
  B = 10
  parser = argparse.ArgumentParser()
  parser.add_argument("model_type", type=str, help='type of models')
  args = parser.parse_args()
  model_type = args.model_type
  datasets = ['tomita_3', 'tomita_4', 'tomita_7', \
              DataSet.BP, DataSet.MR, DataSet.IMDB, DataSet.MNIST, DataSet.TOXIC]
  states_i = np.zeros((B, len(datasets), 5)) # 5:=num_k

  for i in range(B):
    print("========= use bootstrap sample {} for training data =========".format(i))
    for ds_idx, dataset in enumerate(datasets):
      for k_idx, k in enumerate(range(2, 12, 2)):
        pfa_dir = get_path(AbstractData.L2.format(model_type, dataset, i, k, 64))
        pfa_path = glob.glob(get_path(os.path.join(pfa_dir, "*.pm")))[0]
        with open(pfa_path, "r") as fr:
          raw_pm_lines = fr.readlines()
          ptn = re.compile(r".*\[1\.\.(\d+)\].*")
          # pmファイルの4行目は s:[1..5] init 1; のようになっているのでここから状態数(=5)の部分だけ取り出す
          total_states = int(ptn.match(raw_pm_lines[3]).group(1))
          states_i[i][ds_idx][k_idx] = total_states
  boot_mean = np.ceil(np.mean(states_i, axis=0)).astype('int64') # 状態数は整数で示したいので平均した後に切り上げ
  os.makedirs(f'./pfa_states/{model_type}', exist_ok=True)
  with open(f'./pfa_states/{model_type}/boot_avg.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(boot_mean)