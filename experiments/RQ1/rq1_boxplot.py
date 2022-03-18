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
model_type = ModelType.LSTM
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データ集計部分
dict4boxplot = defaultdict(list)
for ngram_n in range(1, 6):
    dataset_susp_scores = list()
    for d, dataset in enumerate(['tomita_3', 'tomita_4', 'tomita_7', \
                  DataSet.BP, DataSet.MR, DataSet.IMDB]):
        dataset_susp_score = list()
        for k in range(2, 12, 2):
            for i in range(B):
                # ngramごとの疑惑値の辞書を取得
                susp_ngram_dict = get_susp_ngram_dict(model_type, dataset, i, k, ngram_n)
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
plt.rcParams["font.size"] = 22 # 全体のフォントサイズが変更されます。
plt.rcParams['xtick.labelsize'] = 22 # 軸だけ変更されます。
plt.rcParams['ytick.labelsize'] = 22 # 軸だけ変更されます
plt.rcParams["figure.subplot.left"] = 0.05  
plt.rcParams["figure.subplot.bottom"] = 0.05
plt.rcParams["figure.subplot.right"] =0.95  
plt.rcParams["figure.subplot.top"] = 0.95


fig = plt.figure(figsize=(24, 12))
for ngram_n in range(1, 6):
    ax = fig.add_subplot(2, 3, ngram_n)
    ax.set_title('$n$={}'.format(ngram_n), fontsize=24)
    ax.set_ylim(0, 1)
    ax.boxplot(dict4boxplot[ngram_n], labels=['Tomita3', 'Tomita4', 'Tomita7', 'BP', 'RTMR', 'IMDB'])
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45)
plt.subplots_adjust(left=0.05, right=0.95, wspace=0.2, hspace=0.4)
plt.savefig('./rq1.pdf', format='pdf', dpi=300, bbox_inches='tight')