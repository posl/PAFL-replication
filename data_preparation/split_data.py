import os
import sys 
# 異なる階層のutilsをインポートするために必要
sys.path.append("../../")
import numpy as np
import random
import math
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_pickle

def data_split4extract(data, data_source, ratio):
  """
  データセットの訓練もしくはテストデータから, 疑惑値に基づいた抽出の対象となるデータ(valデータ;検証データ)を確保する.
  
  Parameters
  ------------------
  data: dict
    processed dataからロードしてきたデータのdict型変数.
    train_x, train_y, test_x, test_yなどのキーを持つ.
  data_source: str
    trainもしくはtestのどちらから検証データを取り出すか. 
  ratio: float
    data_sourceで設定したデータから, 何割のデータを検証データとするか.
    すべてを対象とする場合は1.0. 40%にしたい場合は0.4.
  
  Returns
  ------------------
  data: dict
    入力のdictに対し, ex_x, ex_yというキーを付け加えたもの.
  """
  target_x, target_y = "{}_x".format(data_source), "{}_y".format(data_source)

  # [0, データサイズ)のランダムなindicesを, データサイズ*ratioだけ作成
  l = len(data[target_y])
  size = math.floor(l * ratio)
  indices = random.sample(range(l), k=size)

  # 作成したindicesのデータをex_x, ex_yのキーに代入する
  data["val_x"], data["val_y"] = np.array(data[target_x])[indices], np.array(data[target_y])[indices]
  # 型をlistに戻しておく
  data["val_x"], data["val_y"] = data["val_x"].tolist(), data["val_y"].tolist()

  # {data_source}_x, {data_source}_yの更新(ex_x, ex_yの分を取り除く)
  data[target_x], data[target_y] = np.delete(data[target_x], indices), np.delete(data[target_y], indices)
  # 型をlistに戻しておく
  data[target_x], data[target_y] = data[target_x].tolist(), data[target_y].tolist()

  return data

if __name__ == "__main__":
  # コマンドライン引数から受け取り
  dataset = sys.argv[1]
  # データをロード
  data = load_pickle(get_path(getattr(DataPath, dataset.upper()).PROCESSED_DATA))
  # 抽出対象データをテストデータから4割確保する
  data = data_split4extract(data, data_source="test", ratio=0.4)
  # valデータを分けた後のものにprocessed_dataを更新
  save_pickle(get_path(getattr(DataPath, dataset.upper()).SPLIT_DATA), data)