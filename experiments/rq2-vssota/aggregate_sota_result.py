import os
import sys, glob
import argparse
# 異なる階層のutilsをインポートするために必要
sys.path.append("../")
import numpy as np
import math
import re
import random
import torch
from collections import defaultdict
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_pickle, load_json, save_json
# 異なる階層のmodel_utilsからインポート
from model_utils.model_util import load_model, sent2tensor
from model_utils import train_args

if __name__=='__main__':
  B = 10
  # コマンドライン引数から受け取り
  parser = argparse.ArgumentParser()
  parser.add_argument("model_type", type=str, help='type of models')
  parser.add_argument("--start_boot_id", type=int, help='What boo_id starts from.', default=0)
  args = parser.parse_args()
  model_type, start_boot_id = args.model_type, args.start_boot_id
  
  datasets = ['3', '4', '7', 'bp', 'mr', 'imdb', 'mnist', 'toxic']
  # metrics = ['accuracy', 'precision', 'recall', 'f1', 'mcc']
  metrics = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc']
  for theta in [10, 50]:
    # 各bootでの抽出データの予測結果を入れる箱
    pred_res_dfs = np.ndarray((B, len(datasets), len(metrics)))
    # bootの平均のsave_path
    avg_save_path = get_path(f'comparison/artifacts/{model_type}/pred_result_for_look/theta={theta}/boot_avg.csv')
    for i in range(start_boot_id, B):
      save_path = get_path(f'comparison/artifacts/{model_type}/pred_result_for_look/theta={theta}/boot_{i}.csv')
      for j, dataset in enumerate(datasets):
        npz_path = get_path(f'comparison/artifacts/{model_type}/{dataset}/boot_{i}/pred_res_theta={theta}.npz')
        pred_res_npz = np.load(npz_path, allow_pickle=True)
        for k, m in enumerate(metrics):
          pred_res_dfs[i][j][k] = pred_res_npz[m] if str(pred_res_npz[m]) != 'None' else np.nan
      os.makedirs(os.path.dirname(save_path), exist_ok=True)
      np.savetxt(save_path, pred_res_dfs[i], delimiter=',', fmt='%.3f')
      print(f'saved in {save_path}')
    avg_pred_res = np.mean(pred_res_dfs, axis=0)
    os.makedirs(os.path.dirname(avg_save_path), exist_ok=True)
    np.savetxt(avg_save_path, avg_pred_res, delimiter=',', fmt='%.3f')
    print(f'saved in {avg_save_path}')
