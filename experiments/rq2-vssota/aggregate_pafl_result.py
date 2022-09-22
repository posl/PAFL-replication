import os
import sys, glob
import argparse
# 異なる階層のutilsをインポートするために必要
sys.path.append("../")
import numpy as np
import math
import re
import random
from collections import defaultdict
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_pickle, load_json, save_json
# 異なる階層のmodel_utilsからインポート
from model_utils.model_util import load_model, sent2tensor
from model_utils import train_args

if __name__=='__main__':
  # コマンドライン引数から受け取り
  parser = argparse.ArgumentParser()
  parser.add_argument("model_type", type=str, help='type of models')
  parser.add_argument("theta", type=int, help='the threshold of extraction', choices=[10, 50])
  args = parser.parse_args()
  model_type, theta = args.model_type, args.theta

  method_names = ['ochiai', 'tarantula', 'dstar', 'ochiai2', 'ample']
  num_dataset = 8
  k_list, n_list = list(range(2, 12, 2)), list(range(1, 6, 1))
  num_k, num_n = len(k_list), len(n_list)
  
  dataset_res_per_n = np.zeros((num_n, num_dataset, 6))

  for method in method_names:
    for ni, n in enumerate(n_list):
      pafl_path = get_path(f'data/extracted_data/{model_type}/pred_table_for_look/{method}/boot_avg/n={n}_theta={theta}.csv')
      pafl_res = np.genfromtxt(pafl_path, delimiter=',', dtype='float')
      pafl_res_ex = pafl_res[:,::2].astype(np.float32) # loadした直後のshape: (40, 12)を[:,::2]で(40, 6)にする(ランダムのを落とすため)
      # print(pafl_res_ex, pafl_res_ex.shape)
      for dataset_id, i in enumerate(range(0, num_dataset*num_k, num_k)):
        dataset_res = pafl_res_ex[i:i+num_k, :]
        # print(dataset_res.shape) #(num_k, 6)
        dataset_res = np.where(dataset_res < 0, np.nan, dataset_res)
        # theta, n, datasetを固定したときの全k(5通り)の中の最小値を取る
        dataset_res_k_aggregated = np.nanmin(dataset_res, axis=0)
        # print(dataset_res_k_aggregated.shape) #(6, )
        dataset_res_per_n[ni][dataset_id] = dataset_res_k_aggregated
    # theta,datasetを固定したときの全n, kの中の最小値を取る
    pafl_res_aggregated = np.nanmin(dataset_res_per_n, axis=0)
    # print(pafl_res_aggregated.shape) # (8, 6) = (#datasets, #metrics)
    # 保存パスを指定して保存
    save_path = get_path(f'comparison/artifacts/{model_type}/{method}/pafl_best_res_theta={theta}.csv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savetxt(save_path, pafl_res_aggregated, delimiter=',', fmt='%.3f')
    print(f'saved in {save_path}')