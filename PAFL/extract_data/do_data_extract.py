import os
import sys 
import argparse
# 異なる階層のutilsをインポートするために必要
sys.path.append("../../")
import numpy as np
import math
import re
import random
import time
from collections import defaultdict
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_pickle, load_json, save_json
from utils.susp_score import *

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

def random_indices(data_size, theta):
  # 抽出するデータ数
  extract_size = math.floor(data_size * theta / 100)
  # [0, data_size)の範囲で, extract_sizeと同数のランダムな乱整数列を生成
  return random.sample(range(data_size), k=extract_size)

def extract_processed_data(boot, ex_indices, rand_indices, key="val"):
  """
  指定されたデータセットをロードする.
  データセットの辞書型変数の, キー{key}_x, {key}_yが抽出の対象となる.
  指定したキーのデータからex_indices, rand_indicesのそれぞれで切り出す.
  切り出したx, yを辞書型変数にまとめて返す.
  
  Parameters
  ------------------
  boot: dict
    bootstrap samplingしたデータ(抽出の対象となるもの). 
  ex_indices: list of int
    抽出データを抽出するためのindices.
  rand_indices: list of int
    ランダムデータを抽出するためのindices.
  key: str, default='val'
    データセットの辞書型変数のうち, どのキーのデータを抜き出すか. 
  
  Returns
  ------------------
  ex_data: dict
    抽出データの辞書. 
    x, yというキーにそれぞれサンプル, ラベルを格納する.
  rand_data: dict
    ランダムデータの辞書. 
    x, yというキーにそれぞれサンプル, ラベルを格納する.
  """
  target_x, target_y = "{}_x".format(key), "{}_y".format(key)
  all_target_x, all_target_y = np.array(boot[target_x]), np.array(boot[target_y])

  # 与えられたindicesを用いてデータを取り出す
  ex_x, ex_y = all_target_x[ex_indices].tolist(), all_target_y[ex_indices].tolist()
  rand_x, rand_y = all_target_x[rand_indices].tolist(), all_target_y[rand_indices].tolist()
  
  # exデータ, randデータをそれぞれdictにまとめて返す
  ex_data, rand_data = {}, {}

  # 予測で使うためにはx, y以外のキーも設定しないといけないので
  ex_data["x"], ex_data["y"] = ex_x, ex_y
  rand_data["x"], rand_data["y"] = rand_x, rand_y  
  keys = list(boot.keys()).copy()
  keys.remove('val_x'); keys.remove('val_y')
  keys.remove('test_x'); keys.remove('test_y')
  for key in keys:
    ex_data[key] = boot[key]
    rand_data[key] = boot[key]
  return ex_data, rand_data

def make_extracted_data_info(model_type, dataset, boot_id, k, ngram_n, theta, ex_data, rand_data):
  """
  ex_dataやrand_dataの情報をまとめたjsonファイルを作成する
  
  Parameters
  ------------------
  model_type: str
    RNNモデルのタイプ.
  dataset: str
    データセットのタイプ.
  boot_id: int
    何番目のbootstrap samplingを使うか．
  k: int
    pfaのクラスタ数.
  ngram_n: int
    ngramのn.
  theta: int
    疑惑値の通過回数の上位何%を抽出するかのパラメータ.
  ex_data: dict
    疑惑値に着目して抽出したデータ
  rand_data: dict
    疑惑値に着目して抽出したデータと同数のランダムに取り出したデータ
  """
  data_info = {}
  data_info["dataset"] = dataset
  data_info["model_type"] = model_type
  data_info["k"] = k
  data_info["ngram_n"] = ngram_n
  data_info["theta"] = theta
  data_info["data_size"] = len(ex_data["x"])
  # ex, rand各データの, 各ラベルのサンプル数
  data_info["ex"], data_info["rand"] = {}, {}
  data_info["ex"]["1"] = sum(ex_data["y"])
  data_info["ex"]["0"] = len(ex_data["y"]) - data_info["ex"]["1"]
  data_info["rand"]["1"] = sum(rand_data["y"])
  data_info["rand"]["0"] = len(rand_data["y"]) - data_info["rand"]["1"]
  save_json(os.path.join(ExtractData.DIR.format(model_type, dataset, boot_id, k, ngram_n, theta), "extracted_data_info.json"), data_info)

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
  # 抽出対象となるデータ
  target_source = "val"
  # pfa tracesが保存されているファイル名
  pfa_trace_name = "{}_by_train_partition.txt".format(target_source)
  print("dataset = {}\nmodel_type= {}".format(dataset, model_type))
  
  elapsed_iknth = np.zeros((B, 5, 5, 2))

  for i in range(start_boot_id, B):
    print("\n========= boot_id={} =========".format(i))

    # bootstrap samplingをロード
    boot = load_pickle(get_path(os.path.join(getattr(DataPath, dataset.upper()).BOOT_DATA_DIR, "{}_boot_{}.pkl".format(target_source, i)))) if not isTomita else \
      load_pickle(get_path(os.path.join(DataPath.TOMITA.BOOT_DATA_DIR.format(tomita_id), "{}_boot_{}.pkl".format(target_source, i))))

    for ki, k in enumerate(range(start_k, 12, 2)):
      print("========= k={} =========".format(k))      
      # pfaトレースの保存パスを取得
      pfa_trace_path = os.path.join(AbstractData.PFA_TRACE.format(model_type, dataset, i, k), pfa_trace_name)

      for ni, n in enumerate(range(start_n, 6)):
        print("========= n={} =========".format(n))
        
        theta2array = defaultdict(list) # theta(={10,50})から抽出データのインデックスへの辞書, valのshapeは(#method, #抽出データ)
        for method in method_names:
          # ngramごとの疑惑値の辞書を取得
          susp_ngram_dict = get_susp_ngram_dict(model_type, dataset, i, k, n, method)
          # print(susp_ngram_dict)
          # exit()
          # ans scoreのdictを得る
          ans_score = score_relative_ngram_susp(susp_ngram_dict, n, pfa_trace_path)

          # thetaを10から50まで10刻み
          # for theta in range(10, 60, 10):
          for thetai, theta in enumerate([10, 50]):
            print("========= theta={} =========".format(theta))
            # スコアの上位theta%のデータのインデックス及びランダムなインデックスを取得
            s_time = time.perf_counter()
            ex_indices = extract_indices(ans_score, theta)
            f_time = time.perf_counter()
            elapsed_iknth[i][ki][ni][thetai] = f_time - s_time
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
          rand_indices = random_indices(len(boot['val_x']), theta)
          os.makedirs(ExtractData.DIR.format(model_type, dataset, i, k, n, theta), exist_ok=True)
          np.savez(os.path.join(ExtractData.DIR.format(model_type, dataset, i, k, n, theta), 'names_indices'), \
            names=np.array(method_names), indices=np.array(theta2array[theta]), rand=rand_indices)
  # データセットごとの平均時間を保存
  elapsed_mean = np.mean(elapsed_iknth)
  print(f'mean elapsed in {B} boot, five k settings, five n settings, two thetas: {elapsed_mean} [sec.]')
  with open(f'elapsed_time_{model_type}.csv', 'a') as f:
    f.write(f'{dataset}, {elapsed_mean}\n')