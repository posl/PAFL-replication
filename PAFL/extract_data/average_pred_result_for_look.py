"""
n,thetaごとに全bootの予測結果を平均したcsvを吐き出す．
あらかじめmake_pred_result_for_look.pyを実行しておいて，bootごとの予測をまとめたcsvを取得しておくこと．
"""
import os
import sys 
import argparse
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
  args = parser.parse_args()
  model_type = args.model_type

  for method in method_names:
    # 各bootについてとりあえず平均する
    for n in range(1, 6):
      for theta in [10, 50]:
        save_path = get_path(f"data/extracted_data/{model_type}/pred_table_for_look/{method}/boot_avg/n={n}_theta={theta}.csv")
        pred_res_dfs = np.ndarray((B, num_k*len(datasets), num_metrics*num_mode))
        for i in range(B):
          # pred_resの保存パスを取得
          pred_res_save_path = ExtractData.PRED_RESULT_FOR_LOOK.format(model_type, method, i, n, theta)
          pred_res_df = pd.read_csv(pred_res_save_path)
          # 各bootでの予測メトリクスを代入
          pred_res_dfs[i] = pred_res_df.drop(columns=['dataset', 'k']).to_numpy()
        # boot全体で平均
        boot_mean = np.nanmean(pred_res_dfs, axis=0) # shape: (num_k*len(datasets), num_metrics*num_mode)
        # n, thetaごとにcsvで保存する
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savetxt(save_path, boot_mean, delimiter=',', fmt='%.3f')

        # 平均した結果の加工
        # # ランダムデータに対する精度をkによって買えないようにする
        # # aucが不定の部分の対処
        boot_mean_csv = pd.read_csv(save_path, header=None)
        # ランダムに対する精度は，k=2の時のものを用いることにする．
        nrow, ncol = boot_mean_csv.shape
        for c in range(1, ncol, 2):
          for r in range(nrow):
            if r % 5 == 0:
              continue
            else:
              boot_mean_csv.iat[r, c] = '-'
        boot_mean_csv.to_csv(save_path, header=False, index=False, sep=",")
        print(f'saved in {save_path}')
        