import os
import sys 
import csv
import numpy as np
import math
import random
import argparse
from collections import defaultdict
import glob 
import torch
import matplotlib.pyplot as plt
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from utils.time_util import current_timestamp
from utils.help_func import filter_stop_words, load_pickle, save_pickle, load_json, save_json
from utils.constant import *
from utils.susp_score import *
from model_utils.model_util import load_model
# ゼロ除算を防ぐための定数
EPS = 1e-09


def extract_indices(score_dict, theta):
  """
  各データの疑惑状態の通過回数から, 疑惑状態の通過回数が上位theta%だったデータのインデックスのリストを返す.
  
  Parameters
  ------------------
  score_dict: defaultdict(float)
    pfaトレースの行番号 -> トレース(データサンプル)に対するスコアの値 の対応辞書.
  theta: int
    疑惑値の通過回数の上位何%を抽出するかのパラメータ.
  
  Returns
  ------------------
  ex_indices: list of int
    疑惑状態の通過回数が上位theta%だったデータのインデックスのリスト.
  """
  # 対象となるデータ数
  data_size = len(score_dict)
  # 抽出するデータ数
  extract_size = math.floor(data_size * theta / 100)
  # スコアの降順でソートする
  score_dict = sorted(score_dict.items(), key=lambda x:x[1], reverse=True)
  # スコアの上位theta%のデータだけ切り出す
  score_dict = score_dict[:extract_size]

  ex_indices = []
  # 疑惑状態の高い順からextract_size個のデータの行番号を追加していく
  for e in score_dict:
    ex_indices.append(e[0])
  return ex_indices

if __name__ == "__main__":
  B = 10
  device = "cpu"
  # コマンドライン引数から受け取り
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", type=str, help='abbrev. of datasets')
  parser.add_argument("model_type", type=str, help='type of models')
  parser.add_argument("--start_boot_id", type=int, help='What boo_id starts from.', default=0)
  parser.add_argument("--start_k", type=int, help='What k starts from.', default=2)
  parser.add_argument("--start_n", type=int, help='What n starts from.', default=1)
  args = parser.parse_args()
  dataset, model_type = args.dataset, args.model_type
  start_boot_id, start_k, start_n = args.start_boot_id, args.start_k, args.start_n
  
  method_names = ['ochiai', 'tarantula', 'dstar', 'ochiai2', 'ample']

  isTomita = dataset.isdigit()
  # 変数datasetを変更する(tomitaの場合，"1" => "tomita_1"にする)
  if isTomita:
    tomita_id = dataset
    dataset = "tomita_" + tomita_id
  isImage = (dataset == DataSet.MNIST)
  input_type = 'text' if not isImage else 'image'
  num_class = 10 if isImage else 2

  for i in range(start_boot_id, B):
    print("\n========= boot_id={} =========".format(i))

    # bootstrap samplingをロード
    boot = load_pickle(get_path(os.path.join(getattr(DataPath, dataset.upper()).BOOT_DATA_DIR, "val_boot_{}.pkl".format(i)))) if not isTomita else \
      load_pickle(get_path(os.path.join(DataPath.TOMITA.BOOT_DATA_DIR.format(tomita_id), "val_boot_{}.pkl".format(i))))

    for k in range(start_k, 12, 2):
      print("========= k={} =========".format(k))      
      # dfaトレースの保存パスを取得
      dfa_trace_path = get_path(f'data/dfa/{model_type}/{dataset}/boot_{i}/k={k}/val_trace.txt')

      for n in range(start_n, 6):
        print("========= n={} =========".format(n))
        save_dir = get_path(f'data/dfa/{model_type}/{dataset}/boot_{i}/k={k}/n={n}')
        
        theta2array = defaultdict(list) # theta(={10,50})から抽出データのインデックスへの辞書, valのshapeは(#method, #抽出データ)
        for method in method_names:
          print("========= method={} =========".format(method))
          # ngramごとの疑惑値の辞書を取得
          susp_ngram_dict = load_pickle(get_path(f'data/dfa/{model_type}/{dataset}/boot_{i}/k={k}/n={n}/{method}_susp_dict.pkl'))
          # dictのkeyだけtuple=>str型にして辞書を再構成する
          susp_ngram_dict = dict(list(
              map(lambda x : (str(x[0]),x[1]), susp_ngram_dict.items())
          ))
          # ans scoreのdictを得る
          ans_score = score_relative_ngram_susp(susp_ngram_dict, n, dfa_trace_path)

          for theta in [10, 50]:
            print("========= theta={} =========".format(theta))
            # スコアの上位theta%のデータのインデックス及びランダムなインデックスを取得
            ex_indices = extract_indices(ans_score, theta)  
            theta2array[theta].append(ex_indices)
            # インデックスからデータの取り出しを行う
            # ex_data, rand_data = extract_processed_data(boot, ex_indices, rand_indices)
            # 取り出したデータの情報をjsonで保存
            # make_extracted_data_info(model_type, dataset, i, k, n, theta, ex_data, rand_data)
            # 抽出したデータをpklで保存する
            # save_pickle(ExtractData.EX.format(model_type, dataset, i, k, n, theta), ex_data)
            # save_pickle(ExtractData.RAND.format(model_type, dataset, i, k, n, theta), rand_data)
        
        # 結果を保存
        for theta in [10, 50]:
          np.savez(os.path.join(save_dir, f'names_indices_theta={theta}'), \
            names=np.array(method_names), indices=np.array(theta2array[theta]))
          print('saved in ' +  f'{os.path.join(save_dir, f"names_indices_theta={theta}")}')