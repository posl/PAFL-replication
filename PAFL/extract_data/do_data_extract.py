import os
import sys 
# 異なる階層のutilsをインポートするために必要
sys.path.append("../../")
import numpy as np
import math
import re
import random
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
  rand_indices: list of int
    ex_indicesと同じ数だけのインデックスのリスト.
  """
  # 対象となるデータ数
  data_size = len(score_dict)
  # 抽出するデータ数
  extract_size = math.floor(data_size * theta / 100)
  # スコアの降順でソートする
  score_dict = sorted(score_dict.items(), key=lambda x:x[1], reverse=True)
  # スコアの上位theta%のデータだけ切り出す
  score_dict = score_dict[:extract_size]

  ex_indices, rand_indices = [], []
  # 疑惑状態の高い順からextract_size個のデータの行番号を追加していく
  for e in score_dict:
    ex_indices.append(e[0])
  # [0, data_size)の範囲で, extract_sizeと同数のランダムな乱整数列を生成
  rand_indices = random.sample(range(data_size), k=extract_size)
  
  return ex_indices, rand_indices

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
  ex_data["vocab"], ex_data["classes"], ex_data["word_to_idx"], ex_data["idx_to_word"] = \
    boot["vocab"], boot["classes"], boot["word_to_idx"], boot["idx_to_word"]

  rand_data["x"], rand_data["y"] = rand_x, rand_y
  rand_data["vocab"], rand_data["classes"], rand_data["word_to_idx"], rand_data["idx_to_word"] = \
    boot["vocab"], boot["classes"], boot["word_to_idx"], boot["idx_to_word"]

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
  # コマンドライン引数から受け取り
  dataset = sys.argv[1]
  model_type = sys.argv[2]
  isTomita = dataset.isdigit()
  # 変数datasetを変更する(tomitaの場合，"1" => "tomita_1"にする)
  if isTomita:
    tomita_id = dataset
    dataset = "tomita_" + tomita_id
  # sbflのメソッドはochiaiで固定とする
  method = "ochiai"
  # 抽出対象となるデータ
  target_source = "val"
  # pfa tracesが保存されているファイル名
  pfa_trace_name = "{}_by_train_partition.txt".format(target_source)
  print("dataset = {}\nmodel_type= {}".format(dataset, model_type))
  
  # オリジナルのデータの読み込み
  ori_data = load_pickle(get_path(getattr(DataPath, dataset.upper()).SPLIT_DATA)) if not isTomita else \
    load_pickle(get_path(DataPath.TOMITA.SPLIT_DATA.format(tomita_id)))

  for i in range(B):
    print("\n========= boot_id={} =========".format(i))

    # bootstrap samplingをロード
    boot = load_pickle(get_path(os.path.join(getattr(DataPath, dataset.upper()).BOOT_DATA_DIR, "{}_boot_{}.pkl".format(target_source, i)))) if not isTomita else \
      load_pickle(get_path(os.path.join(DataPath.TOMITA.BOOT_DATA_DIR.format(tomita_id), "{}_boot_{}.pkl".format(target_source, i))))
    # train_x, train_y以外の属性は合わせる
    boot["vocab"], boot["classes"], boot["word_to_idx"], boot["idx_to_word"] = \
      ori_data["vocab"], ori_data["classes"], ori_data["word_to_idx"], ori_data["idx_to_word"]

    for k in range(2, 12, 2):
      print("========= k={} =========".format(k))      
      # pfaトレースの保存パスを取得
      pfa_trace_path = os.path.join(AbstractData.PFA_TRACE.format(model_type, dataset, i, k), pfa_trace_name)

      for n in range(1, 6):
        print("========= n={} =========".format(n))
        # ngramごとの疑惑値の辞書を取得
        susp_ngram_dict = get_susp_ngram_dict(model_type, dataset, i, k, n)
        # relative_ngram_susp_scoreのdictを得る
        relative_ngram_susp_score = score_relative_ngram_susp(susp_ngram_dict, n, pfa_trace_path)

        # thetaを10から50まで10刻み
        # for theta in range(10, 60, 10):
        for theta in [10, 50]:
          print("========= theta={} =========".format(theta))
          # スコアの上位theta%のデータのインデックス及びランダムなインデックスを取得
          ex_indices, rand_indices = extract_indices(relative_ngram_susp_score, theta)
          # インデックスからデータの取り出しを行う
          ex_data, rand_data = extract_processed_data(boot, ex_indices, rand_indices)
          # 取り出したデータの情報をjsonで保存
          make_extracted_data_info(model_type, dataset, i, k, n, theta, ex_data, rand_data)
          # 抽出したデータをpklで保存する
          save_pickle(ExtractData.EX.format(model_type, dataset, i, k, n, theta), ex_data)
          save_pickle(ExtractData.RAND.format(model_type, dataset, i, k, n, theta), rand_data)