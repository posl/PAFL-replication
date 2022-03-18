import os
import sys
import time
sys.path.append("../../") # utils, model_utilsをインポートするのに必要
from os.path import abspath
import copy
# 異なる階層のmodel_utilsからインポート
from model_utils import train_args
from model_utils.model_util import save_model, add_data_info
from model_utils.trainer import train, test
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_readme

# コマンドライン引数からデータセットやモデルタイプなどを取得
dataset = sys.argv[1]
model_type = sys.argv[2]
# tomitaかどうか判定
isTomita = True if dataset.isdigit() else False
gpu = int(sys.argv[3])

# use_cleanはストップワード除去を行うかどうかのフラグ
# 自然言語のデータセットの場合のみストップワード除去を行う
# カッコがストップワードに入っているのでBPに関しては除去してはいけない
if dataset == DataSet.MR or dataset == DataSet.IMDB:
  use_clean = 1
else:
  use_clean = 0

# train_args.pyからハイパーパラメータを取得
params = getattr(train_args, "args_{}_{}".format(model_type, dataset))() if not isTomita else \
  getattr(train_args, "args_{}_{}".format(model_type, "tomita"))()
# データセットと埋め込み行列をロード
data = load_pickle(get_path(getattr(DataPath, dataset.upper()).SPLIT_DATA)) if not isTomita else \
  load_pickle(get_path(DataPath.TOMITA.SPLIT_DATA.format(dataset)).format(dataset))
wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX)) if not isTomita else \
  load_pickle(get_path(DataPath.TOMITA.WV_MATRIX.format(dataset)).format(dataset))
# データや埋め込み行列の情報をparamsに追加
add_data_info(data, params)
params["WV_MATRIX"] = wv_matrix
params["device"] = "cuda:{}".format(gpu) if gpu >= 0 else "cpu"
# params["device"] = 'cpu'
params["rnn_type"] = model_type
params["use_clean"] = use_clean

# bootstrap sampingの回数
B = 10
# 各bootstrapを使った訓練の実行時間を保存しておくための配列
executed_time = []
# 各bootstrap sampleで学習したモデルの test accuracyの配列
models_acc = []
# 各bootstrap sampleについて繰り返し
for i in range(B):
  
  print("========= use bootstrap sample {} for training data =========".format(i))
  boot = load_pickle(get_path(os.path.join(getattr(DataPath, dataset.upper()).BOOT_DATA_DIR, "boot_{}.pkl".format(i)))) if not isTomita else \
    load_pickle(get_path(os.path.join(DataPath.TOMITA.BOOT_DATA_DIR.format(dataset), "boot_{}.pkl".format(i))))
  # train_x, train_y以外の属性は合わせる
  boot["vocab"], boot["classes"], boot["word_to_idx"], boot["idx_to_word"],  boot["val_x"], boot["val_y"], boot["test_x"], boot["test_y"] = \
      data["vocab"], data["classes"], data["word_to_idx"], data["idx_to_word"], data["val_x"], data["val_y"], data["test_x"], data["test_y"]
    
  # モデルの訓練を行う
  start_time = time.time()
  model, train_acc, test_acc = train(boot, params)
  fin_time = time.time()
  executed_time.append(fin_time - start_time)
  models_acc.append(test_acc)
  print("The training time for {} epochs: {:.2f}sec.".format(params["EPOCH"], executed_time[i]))

  # 学習済みモデルの保存
  save_folder = os.path.join(getattr(getattr(TrainedModel, model_type.upper()), dataset.upper()), "boot_{}".format(i)) if not isTomita else \
    os.path.join(getattr(TrainedModel, model_type.upper()).TOMITA.format(dataset), "boot_{}".format(i))
  save_path = get_path(save_folder)
  # モデルファイルはpklで, train_acc, test_accを名前に付けて保存する
  save_model(abspath(save_path), model, train_acc, test_acc)
  # paramsの情報をreadmeとして保存する
  model_identifier = "train_acc-{:.4f}-test_acc-{:.4f}".format(train_acc, test_acc)
  save_readme(parent_path=save_path,
              content=["{}:{}\n".format(key, params[key]) for key in params.keys() if key != "WV_MATRIX"],
              identifier=model_identifier)
  print("model saved to {}.\n".format(save_path))

# 最後に平均のテスト精度と合計の実行時間を表示
print("The average accuracy of {} models: {:.4f}".format(B, sum(models_acc) / len(models_acc) ))
print("The total time of training for {} epochs in {} bootstrap samples: {:.2f}sec.".format(params["EPOCH"], B, sum(executed_time)))