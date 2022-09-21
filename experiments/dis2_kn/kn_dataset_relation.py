import os
import sys 
import csv
import numpy as np
import pandas as pd
import argparse
import math
import random
from collections import defaultdict
import glob 
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('whitegrid')
# 異なる階層のutilsをインポートするために必要
sys.path.append("../")
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_pickle, load_json, save_json
from utils.susp_score import *
plt.rcParams['xtick.labelsize'] = 22 # 軸だけ変更されます。
plt.rcParams['ytick.labelsize'] = 22 # 軸だけ変更されます
plt.rcParams["axes.labelsize"] = 22 # 軸ラベルのfontsize
def count_ex_res(metric, method, theta=10):
  # プロットに使用する辞書
  low_counter = defaultdict(int)  # metricsが最低の(k,n) => カウントの対応
  high_counter = defaultdict(int) # metricsが最高の(k,n) => カウントの対応
  
  for dataset in dataset_name:
    # print("metric: {}, dataset: {}".format(metric, dataset))
    # bootごとの平均を入れる変数
    boot_avg = defaultdict(int)
    for k in range(2, 12, 2):
      for n in range(1, 6):
        for i in range(B):
          # 予測結果の辞書のロード
          ex_res_path = ExtractData.EX_PRED.format(model_type, dataset, i, k, n, theta, method)
          ex_res = load_pickle(ex_res_path)
          # bootごとの平均なのでBで割ってから加算する
          boot_avg[(k, n, metric)] += ex_res[metric.lower()]/B
    # メトリクスの昇順でソート
    sorted_boot_avg = sorted(boot_avg.items(), key=lambda x:x[1], reverse=False)
    # 最小/最大値を抽出
    min_value = sorted_boot_avg[0][1]
    max_value = sorted_boot_avg[-1][1]
    # AUCの最小値が-1というエラー時の値になってしまうことがあるので，それを回避するためのfor loop.
    if metric == 'AUC' and min_value < 0:
      for _, val in sorted_boot_avg:
        if val >= 0:
          min_value = val
          break
    # メトリクスが最小の(k,n)と最大の(k,n)をそれぞれbad/good_counterに加算
    # 注意: 最小/最大が複数ある場合はそれらも取り出す
    for knm, val in sorted_boot_avg:
      if val == min_value:
        low_counter[knm] += 1
      elif val == max_value:
        high_counter[knm] += 1
  return low_counter, high_counter

def df_ex_counter(low_counter, high_counter, metric):
  # plot用の配列を生成
  x, y, cnt, met = list(), list(), list(), list()
  for key, val in low_counter.items():
    x.extend([key[0]])
    y.extend([key[1]])
    met.extend([key[2]])
    cnt.extend([val])
  df1 = pd.DataFrame({'k': x, 'n': y, 'cnt': cnt, 'type': 'lowest', 'met': met})

  x, y, cnt, met = list(), list(), list(), list()
  for key, val in high_counter.items():
    x.extend([key[0]])
    y.extend([key[1]])
    met.extend([key[2]])
    cnt.extend([val])
  df2 = pd.DataFrame({'k': x, 'n': y, 'cnt': cnt, 'type': 'highest', 'met': met})
  df1 = df1.append(df2)
  return df1


if __name__=='__main__':
  # 定数
  B = 10
  # model_type = ModelType.LSTM
  # コマンドライン引数から受け取り
  parser = argparse.ArgumentParser()
  parser.add_argument("model_type", type=str, help='type of models')
  parser.add_argument("method", type=str, help='type of SBFL formulas')
  args = parser.parse_args()
  model_type, method = args.model_type, args.method
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # データセット名の集合
  dataset_name = ['tomita_3', 'tomita_4', 'tomita_7',\
                  DataSet.BP, DataSet.MR, DataSet.IMDB, DataSet.TOXIC, DataSet.MNIST]
  # メトリクス名の集合
  metrics_name = ['accuracy', 'f1_score', 'AUC', 'mcc', 'precision', 'recall']
  # metrics_name = ['accuracy', 'AUC']
  num_metrics = len(metrics_name)
  df = pd.DataFrame(columns=['k', 'n', 'cnt', 'type', 'met'])
  for met in metrics_name:
    low_counter, high_counter = count_ex_res(met, method)
    df = df.append(df_ex_counter(low_counter, high_counter, met))

  plt.figure()
  g = sns.relplot(x='k', y='n', data=df, style='type', markers={'highest': 'o', 'lowest': 'o'}, col='type', row='met', s=200, color='black', legend=False)
  g.set_titles(col_template="{col_name}", row_template="{row_name}", size=22)
  g.fig.subplots_adjust(wspace=0.1, hspace=0.2)
  # 軸などの設定
  plt.xticks(list(range(2, 12, 2)))
  plt.yticks(list(range(1, 6)))
  g.axes[0][0].set_xlabel('k')
  g.axes[0][1].set_xlabel('k')
  g.axes[0][0].set_ylabel('n')
  g.axes[0][1].set_ylabel('n')
  os.makedirs('./discussion1_figs/', exist_ok=True)
  plt.savefig(f'./discussion1_figs/{model_type}_{method}_kn.pdf', format='pdf', dpi=300)