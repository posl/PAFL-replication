import os
import sys 
import csv
import numpy as np
import math
import random
from collections import defaultdict
import glob 
import torch
import matplotlib.pyplot as plt
import argparse

import seaborn as sns
sns.set()
# 異なる階層のutilsをインポートするために必要
sys.path.append("../../")
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_pickle, load_json, save_json
from utils.susp_score import *
# 異なる階層のmodel_utilsからインポート
from model_utils.model_util import load_model, get_model_file
from model_utils import train_args
from model_utils.trainer import test_bagging

# 定数
B = 10
# コマンドライン引数から受け取り
parser = argparse.ArgumentParser()
parser.add_argument("model_type", type=str, help='type of models')
parser.add_argument("method", type=str, help='type of SBFL formulas')
args = parser.parse_args()
model_type, method = args.model_type, args.method
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データ集計部分
dict4boxplot = defaultdict(list)
for ngram_n in range(1, 6):
    dataset_susp_scores = list()
    for d, dataset in enumerate(['tomita_3', 'tomita_4', 'tomita_7', \
                  DataSet.BP, DataSet.MR, DataSet.IMDB, DataSet.TOXIC, DataSet.MNIST]):
        dataset_susp_score = list()
        for k in range(2, 12, 2):
            for i in range(B):
                # ngramごとの疑惑値の辞書を取得
                susp_ngram_dict = get_susp_ngram_dict(model_type, dataset, i, k, ngram_n, method)
                # pfa traceのパスを取得
                pfa_trace_path = os.path.join(AbstractData.PFA_TRACE.format(model_type, dataset, i, k), "val_by_train_partition.txt")
                # pfa traceのパスとngramごとの疑惑値から，データごとの疑惑スコア（辞書形式）を計算する
                relative_ngram_susp_score = score_relative_ngram_susp(susp_ngram_dict, ngram_n, pfa_trace_path)
                # 疑惑スコアの平均を取る
                dataset_susp_score.append(np.mean(list(relative_ngram_susp_score.values())))
        dataset_susp_scores.append(dataset_susp_score)
    dict4boxplot[ngram_n] = dataset_susp_scores

# プロット部分
plt.rcParams['font.family'] = 'sans-serif' # font familyの設定
plt.rcParams["font.size"] = 18 # 全体のフォントサイズが変更されます。
plt.rcParams['xtick.labelsize'] = 18 # 軸だけ変更されます。
plt.rcParams['ytick.labelsize'] = 18 # 軸だけ変更されます
plt.rcParams["figure.subplot.left"] = 0.05  
plt.rcParams["figure.subplot.bottom"] = 0.05
plt.rcParams["figure.subplot.right"] =0.95  
plt.rcParams["figure.subplot.top"] = 0.95

os.makedirs('./rq1_figs/', exist_ok=True)
fig = plt.figure(figsize=(24, 4))
for ngram_n in [1, 5]:
  ax = fig.add_subplot(1, 2, 1 if ngram_n==1 else 2)
  ax.set_title(f'{model_type.upper()}, {method}, $n$={ngram_n}', fontsize=20)
  ax.set_ylim(0, 1)
  ax.boxplot(dict4boxplot[ngram_n], labels=['Tomita3', 'Tomita4', 'Tomita7', 'BP', 'RTMR', 'IMDB', 'TOXIC', 'MNIST'])
  labels = ax.get_xticklabels()
  plt.setp(labels, rotation=45)
plt.savefig(f'./rq1_figs/rq1_{model_type}_{method}.pdf', format='pdf', dpi=300, bbox_inches='tight')
print(f'saved in ./rq1_figs/rq1_{model_type}_{method}.pdf')