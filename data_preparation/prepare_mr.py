import os
import sys
sys.path.append("../") # utilsをインポートするため必要
import gensim
from sklearn.utils import shuffle
# 同階層data_preparationからのインポート
from data_preparation.preparation_helper import *
# 異なる階層utilsからのインポート
from utils.help_func import save_pickle
from utils.constant import DataPath, get_path


def divide_mr(word_vectors):
  x, y = [], []
  # 処理済みデータの保存先
  save_path = get_path(DataPath.MR.PROCESSED_DATA)
  # 埋め込み行列の保存先
  save_wv_matrix_path = get_path(DataPath.MR.WV_MATRIX)

  # 生データをロード
  pos_path = os.path.join(get_path(DataPath.MR.RAW_DATA), "rt-polarity.pos")
  neg_path = os.path.join(get_path(DataPath.MR.RAW_DATA), "rt-polarity.neg")
  with open(pos_path, "r", encoding="latin-1") as f:
    for line in f:
      # 生データを整形する
      line = clean_data_for_look(line)
      x.append(line.split())
      # ポジティブなので1のラベルを付与
      y.append(1)
  with open(neg_path, "r", encoding="latin-1") as f:
    for line in f:
      # 生データを整形する
      line = clean_data_for_look(line)
      x.append(line.split())
      # ネガティブなので0のラベルを付与
      y.append(0)
  
  # データをシャッフル
  x, y = shuffle(x, y, random_state=2020)
  # 訓練/テストデータの分割,vocabやword2idxの作成などを行う
  data = set_data(x, y)
  # データセットのサンプル中の単語と事前学習済みword2vecから埋め込み行列を作成
  wv_matrix = make_wv_matrix(data, word_vectors)
  
  # dataと埋め込み行列をpklで保存
  save_pickle(save_path, data)
  save_pickle(save_wv_matrix_path, wv_matrix)
  print("saved in {}".format(save_path))
  print("saved in {}".format(save_wv_matrix_path))
  return data