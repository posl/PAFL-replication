import os
import sys
sys.path.append("../") # utilsをインポートするため必要
# 同階層data_preparationからのインポート
from data_preparation.preparation_helper import set_data
from data_preparation.generate_bp import * # bpデータを生成するためのモジュール
# 異なる階層utilsからのインポート
from utils.help_func import save_pickle
from utils.constant import *

# 他の自然言語のデータセットとは異なり，bpは29種類しか文字(語彙数=29)がないので埋め込みベクトルの作成にはone-hotエンコーディングを用いる
def make_wv_matrix_bp(data):
  index_map = {key: i + 1 for i, key in enumerate(string.ascii_lowercase)}
  index_map[''] = 0
  index_map["("] = 27
  index_map[")"] = 28
  wv_matrix = []
  input_dim = 29  # alphabet_size
  for i in range(len(data["vocab"])):
    word = data["idx_to_word"][i]
    vector = [0.] * input_dim
    vector[index_map[word]] = 1.
    wv_matrix.append(vector)
  wv_matrix = np.array(wv_matrix)
  return wv_matrix

def divide_bp():
  n = 44600
  min_dep = 2
  max_dep = 11
  # bpデータを生成
  bp_data, num_pos, num_neg = get_balanced_parantheses_train_set(n, min_dep, max_dep, lengths=None,
                                                                  max_train_samples_per_length=300,
                                                                  search_size_per_length=200)
  # bpデータを変数に追加していく
  X = []
  Y = []
  for word in bp_data:
    x = [w for w in word] if word != "" else ['']
    X.append(x)
    Y.append(int(bp_data[word]))

  # 訓練/テストデータの分割,vocabやword2idxの作成などを行う
  data = set_data(X, Y)
  # データセットのサンプル中の単語から埋め込み行列を作成
  wv_matrix = make_wv_matrix_bp(data)

  # 生成したデータの保存先
  save_path = get_path(DataPath.BP.PROCESSED_DATA)
  # 埋め込み行列の保存先
  save_wv_matrix_path = get_path(DataPath.BP.WV_MATRIX)
  # dataと埋め込み行列をpklで保存
  save_pickle(save_path, data)
  save_pickle(save_wv_matrix_path, wv_matrix)
  print("saved in {}".format(save_path))
  print("saved in {}".format(save_wv_matrix_path))
  return data