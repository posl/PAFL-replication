import os
import sys
sys.path.append("../") # utilsをインポートするため必要
import numpy as np
from collections import defaultdict
import joblib

# 同階層data_preparationからのインポート
from data_preparation.preparation_helper import *
# 異なる階層utilsからのインポート
from utils.help_func import save_pickle, load_pickle, filter_stop_words
from utils.constant import DataPath, get_path

def divide_toxic(word_vectors):
  # 処理済みデータの保存先
  save_path = get_path(DataPath.TOXIC.PROCESSED_DATA)
  # 埋め込み行列の保存先
  save_wv_matrix_path = get_path(DataPath.TOXIC.WV_MATRIX)
  data = {}
  data['train_x'], data['train_y'] = [], []
  data['test_x'], data['test_y'] = [], []
  # RNNRepairの方でダウンロードしてフィルタリングしておいたデータをロード
  train_lst = joblib.load(os.path.join(get_path(DataPath.TOXIC.RAW_DATA), 'data_list/toxic_train.lst'))
  test_lst = joblib.load(os.path.join(get_path(DataPath.TOXIC.RAW_DATA), 'data_list/toxic_test.lst'))
  
  # X_, y_ の配列に入れていく
  for train_label, train_data in train_lst:
    data['train_x'].append(train_data)
    data['train_y'].append(1 if train_label=='pos' else 0)
  for test_label, test_data in test_lst:
    data['test_x'].append(test_data)
    data['test_y'].append(1 if test_label=='pos' else 0)

  # 語彙や単語<->idの対応を作成
  data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["test_x"] for w in sent])))
  data["classes"] = sorted(list(set(data["train_y"])))
  data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
  data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}

  # データセットのサンプル中の単語と事前学習済みword2vecから埋め込み行列を作成
  print('make wv_matrix from vocab of TOXIC dataset')
  wv_matrix = make_wv_matrix(data, word_vectors)
  
  # dataと埋め込み行列をpklで保存
  save_pickle(save_path, data)
  save_pickle(save_wv_matrix_path, wv_matrix)
  print("saved in {}".format(save_path))
  print("saved in {}".format(save_wv_matrix_path))
  return data