import os
import sys
import argparse
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
  B = 10
  # 有意水準を設定
  sig_level1 = 0.05
  sig_level2 = 0.01
  # コマンドライン引数から受け取り
  parser = argparse.ArgumentParser()
  parser.add_argument("model_type", type=str, help='type of models')
  args = parser.parse_args()
  model_type = args.model_type

  datasets = ['tomita_3', 'tomita_4', 'tomita_7', DataSet.BP, DataSet.MR, DataSet.IMDB, DataSet.MNIST, DataSet.TOXIC]
  method_names = ['ochiai', 'tarantula', 'dstar', 'ochiai2', 'ample']
  for method in method_names:
    for n in range(1, 6):
      rows = defaultdict(list)
      for theta in [10, 50]:
        for dataset in datasets:
          for k in range(2, 12, 2):
            # これらの辞書の各キーはメトリクス名，値はbootごとのメトリクス値のリスト
            ex_mets, rand_mets = defaultdict(list), defaultdict(list)
            for mode in ["ex", "rand"]:
              for i in range(B):
                # 統計結果のcsvを保存するディレクトリ名
                file_name = ExtractData.STAT_RESULT.format(model_type, method, n, theta)
                # メトリクスの値をロードする
                if mode == "ex":
                  pred_res = load_pickle(getattr(ExtractData, "{}_PRED".format(mode.upper())).format(
                      model_type, dataset, i, k, n, theta, method
                    ))
                else:
                  # randデータは常にk=2, n=1のものを読んでくる
                  pred_res = load_pickle(getattr(ExtractData, "{}_PRED".format(mode.upper())).format(
                      model_type, dataset, i, 2, 1, theta
                    ))
                for m in ["accuracy", "precision", "recall", "f1_score", "mcc", "auc"]:
                  if mode == "ex":
                    ex_mets[m].append(pred_res[m])
                  else:
                    rand_mets[m].append(pred_res[m])
            
            row = list()
            for m in ["accuracy", "precision", "recall", "f1_score", "mcc", "auc"]:
              # U検定のP値と統計量uを取得
              try:
                p_value = stats.mannwhitneyu(ex_mets[m], rand_mets[m], alternative="less").pvalue
                p_value = round(p_value, 4)
              except ValueError as e:
                print(e)
                print(m, ex_mets[m], rand_mets[m])
                row.append("NaN")
                continue
              u = stats.mannwhitneyu(ex_mets[m], rand_mets[m], alternative="less").statistic
              
              # cliffのdeltaとその評価結果を計算する
              # d_cliff = np.abs(2*u/(len(ex_mets[m])*len(rand_mets[m])) - 1)
              d_cliff = - (2*u/(len(ex_mets[m])*len(rand_mets[m])) - 1)
              d_cliff = round(d_cliff, 4)
              # print("cliff's d = {:.3f}".format(d_cliff), "({})".format(assess_cliffs_d(d_cliff)))
              
              # p値に応じた値をrowに追加する
              if p_value < sig_level2:
                # print("p-value =", p_value, "(p < {})".format(sig_level2))
                row.append(str(d_cliff) + "0"*(5-len(str(d_cliff))) + "**")
              elif p_value < sig_level1:
                # print("p-value =", p_value, "(p < {})".format(sig_level1))
                row.append(str(d_cliff) + "0"*(5-len(str(d_cliff))) + "*")
              else:
                # print("p-value =", p_value, "(p >= {})".format(sig_level1))
                row.append(str(d_cliff) + "0"*(5-len(str(d_cliff))))
            rows[(dataset,k)] = row
    
        # 辞書rowsのvaluesをcsvに書き込む
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            for _, val in rows.items():
                writer.writerow(val)
        print("saved to {}".format(file_name))