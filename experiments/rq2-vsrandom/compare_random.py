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

def build_df(model_type, stat):
  dir_name = 'pred_table_for_look' if not stat else 'stat_result'
  datasets = ['Tomita3', 'Tomita4', 'Tomita7', 'BP', 'RTMR', 'IMDB', 'MNIST', 'TOXIC']
  method_names = ['ample', 'tarantula', 'ochiai', 'ochiai2', 'dstar']
  # 一旦カウントのための辞書を作成
  dic_cnt = defaultdict(defaultdict)
  for ds in datasets:
    dic_cnt[ds] = defaultdict(int)
  # カウントを行う
  for method in method_names:
    # print(f'---------- method = {method} ----------')
    for n in range(1, 6):
      for theta in [10, 50]:
        load_path = get_path(f"data/extracted_data/{model_type}/{dir_name}/{method}/avg_cnt/n={n}_theta={theta}.csv")
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
  variants = ['srnn', 'gru', 'lstm']

  df_lstm = build_df('lstm', False)
  df_gru = build_df('gru', False)
  df_srnn = build_df('srnn', False)
  df_lstm_stat = build_df('lstm', True)
  df_gru_stat = build_df('gru', True)
  df_srnn_stat = build_df('srnn', True)

  tab6_avg, tab6_stat = pd.DataFrame(columns=datasets), pd.DataFrame(columns=datasets)
  tab7_avg, tab7_stat = pd.DataFrame(columns=method_names), pd.DataFrame(columns=method_names)
  
  # for table 6
  for vari, df in zip(variants, [df_srnn, df_gru, df_lstm]):
    df_per_ds = (df[['dataset', 'cnt']].groupby('dataset').sum()*100/1500).T.reindex(columns=datasets)
    tab6_avg.loc[vari] = df_per_ds.values[0]
  print(f'\nTable 6 (avg)\n{tab6_avg}')
  for vari, df in zip(variants, [df_srnn_stat, df_gru_stat, df_lstm_stat]):
    df_per_ds = (df[['dataset', 'cnt']].groupby('dataset').sum()*100/1500).T.reindex(columns=datasets)
    tab6_stat.loc[vari] = df_per_ds.values[0]
  print(f'\nTable 6 (stat)\n{tab6_stat}')
  
  # for table 7
  for vari, df in zip(variants, [df_srnn, df_gru, df_lstm]):
    df_per_me = (df[['method', 'cnt']].groupby('method').sum()*100/2400).T.reindex(columns=method_names)
    tab7_avg.loc[vari] = df_per_me.values[0]
  print(f'\nTable 7 (avg)\n{tab7_avg}')
  for vari, df in zip(variants, [df_srnn_stat, df_gru_stat, df_lstm_stat]):
    df_per_me = (df[['method', 'cnt']].groupby('method').sum()*100/2400).T.reindex(columns=method_names)
    tab7_stat.loc[vari] = df_per_me.values[0]
  print(f'\nTable 7 (stat)\n{tab7_stat}')

  # save the tables
  tab6_avg.to_csv('./Table6-avg.csv')
  print('\nsaved in Table6-avg.csv')
  tab6_stat.to_csv('./Table6-stat.csv')
  print('saved in Table6-stat.csv')
  tab7_avg.to_csv('./Table7-avg.csv')
  print('saved in Table7-avg.csv')
  tab7_stat.to_csv('./Table7-stat.csv')
  print('saved in Table7-stat.csv')