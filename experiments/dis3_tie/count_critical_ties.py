import os
import sys, glob
import argparse
# 異なる階層のutilsをインポートするために必要
sys.path.append("../../")
import numpy as np
import pandas as pd
import math
import torch
import re
import random
from collections import defaultdict
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_pickle, load_json, save_json
from utils.susp_score import *
# 異なる階層のmodel_utilsからインポート
from model_utils.model_util import load_model, sent2tensor
from model_utils import train_args

import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set()
sns.set_context('paper', 1.5)
import warnings
warnings.simplefilter('ignore')

if __name__ == '__main__':
  methods = ['ochiai', 'tarantula', 'dstar', 'ochiai2', 'ample']
  datasets = ['tomita_3', 'tomita_4', 'tomita_7', 'bp', 'mr', 'imdb', 'toxic', 'mnist'] # mnist忘れてるよ

  # コマンドライン引数から受け取り
  parser = argparse.ArgumentParser()
  parser.add_argument("model_type", type=str, help='type of models')
  args = parser.parse_args()
  model_type = args.model_type

  # データセット，boot，メソッド，n, kごとにまとめる
  # bootでは平均したい
  res = np.zeros((len(datasets), 10, len(methods), 5, 5))

  for di, dataset in enumerate(datasets):
    dataset_susp_scores = list()
    isTomita = dataset.startswith('tomita')
    if isTomita:
      tomita_id = dataset[-1]
    isImage = (dataset == DataSet.MNIST)
    input_type = 'text' if not isImage else 'image'
    num_class = 10 if isImage else 2  
    # オリジナルのデータの読み込み
    ori_data = load_pickle(get_path(getattr(DataPath, dataset.upper()).SPLIT_DATA)) if not isTomita else \
      load_pickle(get_path(DataPath.TOMITA.SPLIT_DATA.format(tomita_id)))
    if not isImage:
      # wv_matrixとinput_dimの設定
      wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX)) if not isTomita else \
        load_pickle(get_path(DataPath.TOMITA.WV_MATRIX.format(tomita_id)))
    # input_dimとuse_cleanの設定
    if isTomita:
      input_dim = 3
    elif isImage:
      input_dim = 28
    elif dataset == DataSet.BP:
      input_dim = 29
    else: # dataset == MR or IMDB
      input_dim = 300
    if dataset == DataSet.MR or dataset == DataSet.IMDB or dataset == DataSet.TOXIC:
      use_clean = 1
    else:
      use_clean = 0

    for i in range(10):
      print(f'----- dataset={dataset}, boot={i} -----')
      # ここでvalデータとモデルを読み出して予測成功or失敗の記録をとっておく
      # データを読み出す
      data = load_pickle(get_path(os.path.join(getattr(DataPath, dataset.upper()).BOOT_DATA_DIR, "val_boot_{}.pkl".format(i)))) if not isTomita else \
      load_pickle(get_path(os.path.join(DataPath.TOMITA.BOOT_DATA_DIR.format(tomita_id), "val_boot_{}.pkl".format(i))))
      # train_x, train_y, val_x, val_y以外の属性は合わせる
      keys = list(ori_data.keys()).copy()
      keys.remove('train_x'); keys.remove('train_y')
      keys.remove('val_x'); keys.remove('val_y')
      for key in keys:
        data[key] = ori_data[key]
      
      # モデルを読み出す
      model_dir = os.path.join(getattr(getattr(TrainedModel, model_type.upper()), dataset.upper()), "boot_{}".format(i)) if not isTomita  else \
        os.path.join(getattr(TrainedModel, model_type.upper()).TOMITA.format(tomita_id), "boot_{}".format(i))
      load_model_path = glob.glob(get_path(os.path.join(model_dir, "*.pkl")))[0]
      model = load_model(model_type, dataset, 'cpu', load_model_path) if not isTomita else \
        load_model(model_type, tomita_id, 'cpu', load_model_path)
      model.eval()

      # 予測結果のリストを返すコードをかいちゃうよ〜
      val_pred = [] # モデルが予測したラベル等
      model.eval()
      for sent, c in zip(data['val_x'], data['val_y']):
        if isImage:
          input_tensor = torch.unsqueeze(torch.tensor(sent), 0) / 255 # (1, 28, 28)
        else:
          input_tensor = sent2tensor(sent, input_dim, ori_data["word_to_idx"], wv_matrix, 'cpu') # (1, sentの単語数, input_size)
        input_tensor = input_tensor.to('cpu') # mnistでだけなぜかinput_tensorのdeviceがcpuになってたので追加した
        output, _ = model(input_tensor)
        lasthn = output[0][-1].unsqueeze(0)
        score = model.h2o(lasthn) # => tensor([[Nのスコア，Pのスコア]])みたいに入ってる
        prob = torch.exp(model.softmax(score)) # => tensor([[Nの確率，Pの確率]])に変換する
        # torch.Tensorからnumpy配列に変換
        prob = prob.cpu().data.numpy()
        pred = np.argmax(prob, axis=1)[0] # predはint型
        val_pred.append(pred)
      val_pred = np.array(val_pred)

      # 各valデータサンプルに対して予測が成功したかどうかの記録(boolean)
      pred_succ = (data['val_y'] == val_pred)

      for mi, method in enumerate(methods):
        dataset_susp_score = list()
        for ni, ngram_n in enumerate(range(1, 6)):
          for ki, k in enumerate(range(2, 12, 2)):
            print(f'-----method={method}, n={ngram_n}, k={k}-----')
            # ngramごとの疑惑値の辞書を取得
            susp_ngram_dict = get_susp_ngram_dict(model_type, dataset, i, k, ngram_n, method)
            # pfa traceのパスを取得
            pfa_trace_path = os.path.join(AbstractData.PFA_TRACE.format(model_type, dataset, i, k), "val_by_train_partition.txt")
            # pfa traceのパスとngramごとの疑惑値から，データごとの疑惑スコア（辞書形式）を計算する
            relative_ngram_susp_score = score_relative_ngram_susp(susp_ngram_dict, ngram_n, pfa_trace_path)
            ans_scores = np.array(list(relative_ngram_susp_score.values()))
            uniq_vals, uniq_inverse, uniq_cnt = np.unique(ans_scores, return_inverse=True, return_counts=True)
            # print(len(uniq_vals), max(uniq_inverse)+1) # len(uniq_vals) == max(uniq_inverse)+1
            # ユニークなansスコアの値から，そのスコアの値のデータが[予測成功する数, 失敗する数]の対応辞書を作る
            dic = defaultdict(lambda: [0, 0])
            for data_idx, uniq_idx in enumerate(uniq_inverse):
              if pred_succ[data_idx]:
                dic[uniq_vals[uniq_idx]][0] += 1
              else:
                dic[uniq_vals[uniq_idx]][1] += 1
            # print(dic)
            critical_tie_cnt = 0
            for ans, lst in dic.items():
              if (lst[0] != 0) and (lst[1] != 0):
                critical_tie_cnt += lst[0] + lst[1]
            critical_tie_rate = critical_tie_cnt / len(data['val_y'])
            res[di][i][mi][ni][ki] = critical_tie_rate
  save_pickle(f'./tie_res_{model_type}.pkl', res)