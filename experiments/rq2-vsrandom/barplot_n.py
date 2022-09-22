import os
import sys 
import argparse
import numpy as np
# 異なる階層のutils, model_utilsをインポートするために必要
sys.path.append("../../")
import numpy as np
import pandas as pd
from collections import defaultdict
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context('paper', 1.5)
import warnings
warnings.simplefilter('ignore')
plt.rcParams['font.family'] = 'sans-serif' # font familyの設定
plt.rcParams["font.size"] = 20 # 全体のフォントサイズが変更されます。
plt.rcParams['xtick.labelsize'] = 18 # 軸だけ変更されます。
plt.rcParams['ytick.labelsize'] = 18 # 軸だけ変更されます
plt.rcParams["figure.subplot.left"] = 0
plt.rcParams["figure.subplot.bottom"] = 0
plt.rcParams["figure.subplot.right"] =0.95  
plt.rcParams["figure.subplot.top"] = 0.95


def build_df_per_n(model_type, n):
  # 一旦カウントのための辞書を作成
  dic_cnt = defaultdict(defaultdict)
  for ds in datasets:
    dic_cnt[ds] = defaultdict(int)
  # カウントを行う
  for method in method_names:
    for theta in [10, 50]:
      load_path = get_path(f"data/extracted_data/{model_type}/pred_table_for_look/{method}/avg_cnt/n={n}_theta={theta}.csv")
      avg_cnt = np.loadtxt(load_path, dtype=int, delimiter=',')
      for idx, arr in enumerate(avg_cnt):
        dic_cnt[datasets[idx]][method] += sum(arr)
  # dfにする
  df_cnt = pd.DataFrame(columns=['dataset', 'method', 'cnt'])
  for ds, dic in dic_cnt.items():
    for method, cnt in dic.items():
      df_cnt = df_cnt.append({'dataset':ds, 'method':method, 'cnt':cnt}, ignore_index=True)
  return df_cnt

if __name__ == '__main__':
  datasets = ['Tomita3', 'Tomita4', 'Tomita7', 'BP', 'RTMR', 'IMDB', 'MNIST', 'TOXIC']
  method_names = ['ample', 'tarantula', 'ochiai', 'ochiai2', 'dstar']
  n_list = list(range(1, 6))
  n_sum = []
  df_cnt = pd.DataFrame(columns=['n', 'variant', 'cnt'])
  for model_type in ['srnn', 'gru', 'lstm']:
    dic_cnt = defaultdict(int)
    for n in n_list:
      df = build_df_per_n(model_type, n)
      dic_cnt[n] = df['cnt'].sum()
    # dfにする
    for n, cnt in dic_cnt.items():
      df_cnt = df_cnt.append({'n':n, 'variant':model_type, 'cnt':cnt}, ignore_index=True)

  # draw the barplot
  fig = plt.figure(figsize=(8, 6), facecolor="w")
  ax = fig.add_subplot(1, 1, 1)
  ax.set_title(f'', fontsize=24)
  g = sns.barplot(data=df_cnt, x='n', y='cnt', hue='variant', palette='rocket_r')
  ax.set_xlabel('n', fontsize=24)
  ax.set_ylabel(None)
  labels = ax.get_xticklabels()
  plt.setp(labels, fontsize=24)
  plt.ylim(0, 480*5)
  plt.yticks([0, 2400*0.6, 2400*0.8, 2400], ['0%', '60%', '80%', '100%\n(2400)'])
  handlers , labels = ax.get_legend_handles_labels()
  for n in n_list:
    val_srnn =int( df_cnt[(df_cnt.n == n)][df_cnt[(df_cnt.n == n)].variant == 'srnn']['cnt'])
    val_gru = int(df_cnt[(df_cnt.n == n)][df_cnt[(df_cnt.n == n)].variant == 'gru']['cnt'])
    val_lstm =int( df_cnt[(df_cnt.n == n)][df_cnt[(df_cnt.n == n)].variant == 'lstm']['cnt'])
    # ax.text(n-1-0.275, val_srnn, f'{round(val_srnn/2400,3):.2%}', ha='center', va='bottom', size=10)
    # ax.text(n-1, val_gru, f'{round(val_gru/2400,3):.2%}', ha='center', va='bottom', size=10)
    ax.text(n-1+0.25, 2400*0.82, f'{round(val_lstm/2400,3):.1%}', ha='center', va='bottom', size=18, color=sns.color_palette('rocket_r', 3)[2], fontweight='bold')
  ax.legend(handlers, ['SRNN', 'GRU', 'LSTM'], \
    bbox_to_anchor=(1, 1), borderaxespad=0, fontsize=20, ncol=3)
  plt.savefig('./Fig11-barplot-n.pdf', format='pdf', dpi=300, bbox_inches='tight')
  print('saved in Fig11-barplot-n.pdf')