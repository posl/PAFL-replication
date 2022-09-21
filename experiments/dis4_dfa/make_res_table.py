import os
import sys 
import argparse
import warnings
warnings.simplefilter('ignore')
# 異なる階層のutils, model_utilsをインポートするために必要
sys.path.append("../../")
import numpy as np
import pandas as pd
from collections import defaultdict
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle

if __name__ == "__main__":
  # 定数 metrics名の配列, 列名もこの順番になる
  metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'auc']
  datasets = ['tomita_3', 'tomita_4', 'tomita_7', DataSet.BP, DataSet.MR, DataSet.IMDB, DataSet.TOXIC]
  method_names = ['ochiai', 'tarantula', 'dstar', 'ochiai2', 'ample']

  # コマンドライン引数から受け取り
  parser = argparse.ArgumentParser()
  parser.add_argument("model_type", type=str, help='type of models')
  args = parser.parse_args()
  model_type = args.model_type

  for method in method_names:
    print("\n============== method={} ==============".format(method))
    for i in range(10):
      print("\n============== boot_id={} ==============".format(i))

      for n in range(1, 6):
        for theta in [10, 50]:
          pred_res_save_path = get_path(f'data/dfa/{model_type}/pred_table_for_look/{method}/boot_{i}/n={n}_theta={theta}.csv')
          # pred_res_dfを作成
          pred_res_df = pd.DataFrame(index=[], columns=[])
          # dfの列のリスト,  とりあえず最初の2つはdatasetとkにする
          columns = ['dataset', 'k']
          for dataset in datasets:
            for k in range(2, 12, 2):
              # もろもろのパス
              save_dir = get_path(f'data/dfa/{model_type}/{dataset}/boot_{i}/k={k}/n={n}')
              pred_res_row = {}
              pred_res_row["dataset"] = dataset
              pred_res_row["k"] = str(k)

              # 予測結果のdictを読み出す
              pred_res = load_pickle(os.path.join(save_dir, f'{method}_pred_result_theta={theta}.pkl'))

              for m in metrics:
                # keyがcolumnsに存在しなかったら, 追加する
                if not m in columns:
                  columns.append(m)
                # メトリクスの値(浮動小数点値)を文字列にし,小数点以下3桁のみ取得
                val_s = str(pred_res[m])[:5]
                # 小数点以下の桁数が3桁未満だった場合, 0で埋める
                if (len(val_s) < 5) and (val_s != 'nan'):
                  pad_size = 5 - len(val_s)
                  val_s += "0" * pad_size
                pred_res_row[m] = val_s    
              # 構成したpred_res_rowをdfに追加
              pred_res_df = pred_res_df.append(pred_res_row, ignore_index=True)
          # columnsの順番にdfの列順を指定
          pred_res_df = pred_res_df.reindex(columns=columns)
          # thetaごとにpred_res_dfをcsvで保存する
          os.makedirs(os.path.dirname(pred_res_save_path), exist_ok=True)
          pred_res_df.to_csv(pred_res_save_path, header=True, index=False, sep=",")

  # bootの平均もやっちゃう
  for method in method_names:
    # 各bootについてとりあえず平均する
    for n in range(1, 6):
      for theta in [10, 50]:
        save_path = get_path(f"data/dfa/{model_type}/pred_table_for_look/{method}/boot_avg/n={n}_theta={theta}.csv")
        pred_res_dfs = np.ndarray((10, 5*len(datasets), 6))
        for i in range(10):
          # pred_resの保存パスを取得
          pred_res_save_path = get_path(f"data/dfa/{model_type}/pred_table_for_look/{method}/boot_{i}/n={n}_theta={theta}.csv")
          pred_res_df = pd.read_csv(pred_res_save_path)
          # 各bootでの予測メトリクスを代入
          pred_res_dfs[i] = pred_res_df.drop(columns=['dataset', 'k']).to_numpy()
        # boot全体で平均
        boot_mean = np.nanmean(pred_res_dfs, axis=0) # shape: (num_k*len(datasets), num_metrics*num_mode)
        # n, thetaごとにcsvで保存する
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savetxt(save_path, boot_mean, delimiter=',', fmt='%.3f')

        # # 平均した結果の加工
        # # # ランダムデータに対する精度をkによって買えないようにする
        # # # aucが不定の部分の対処
        # boot_mean_csv = pd.read_csv(save_path, header=None)
        # # ランダムに対する精度は，k=2の時のものを用いることにする．
        # nrow, ncol = boot_mean_csv.shape
        # for c in range(1, ncol, 2):
        #   for r in range(nrow):
        #     if r % 5 == 0:
        #       continue
        #     else:
        #       boot_mean_csv.iat[r, c] = '-'
        # boot_mean_csv.to_csv(save_path, header=False, index=False, sep=",")
        print(f'saved in {save_path}')
        