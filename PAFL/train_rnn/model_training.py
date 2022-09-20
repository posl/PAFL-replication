import os
import sys
import copy
import argparse
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

# コマンドライン引数の受け取り
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help='abbrev. of datasets')
parser.add_argument("model_type", type=str, help='type of models')
parser.add_argument("--device", type=str, help='cpu or id of gpu', default='cpu') # gpuならその番号(0,1,...)
parser.add_argument("--start_boot_id", type=int, help='What id of bootstrap sample starts training from.', default=0)
args = parser.parse_args()

dataset, model_type = args.dataset, args.model_type

# tomitaかどうか判定
isTomita = True if args.dataset.isdigit() else False
# 画像データかどうか判定
isImage = True if args.dataset == DataSet.MNIST else False


# use_cleanはストップワード除去を行うかどうかのフラグ
# 自然言語のデータセットの場合のみストップワード除去を行う
# カッコがストップワードに入っているのでBPに関しては除去してはいけない
if dataset == DataSet.MR or dataset == DataSet.IMDB or dataset == DataSet.TOXIC:
  use_clean = 1
else:
  use_clean = 0

# train_args.pyからハイパーパラメータを取得
params = getattr(train_args, "args_{}_{}".format(model_type, dataset))() if not isTomita else \
  getattr(train_args, "args_{}_{}".format(model_type, "tomita"))()
# データセットをロード
print('load data and wv_matrix....')
data = load_pickle(get_path(getattr(DataPath, dataset.upper()).SPLIT_DATA)) if not isTomita else \
  load_pickle(get_path(DataPath.TOMITA.SPLIT_DATA.format(dataset)).format(dataset))
# 単語の埋め込み行列をロード(画像でないデータに対してだけ)
if not isImage:
  if not isTomita:
    wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX))
  else:
    wv_matrix = load_pickle(get_path(DataPath.TOMITA.WV_MATRIX.format(dataset)).format(dataset))
  # データや埋め込み行列の情報をparamsに追加
  add_data_info(data, params)
  params["WV_MATRIX"] = wv_matrix
params["device"] = "cuda:{}".format(args.device) if args.device.isdigit() else "cpu"
params["rnn_type"] = model_type
params["use_clean"] = use_clean
params["is_image"] = isImage

# bootstrap sampingの回数
B = 10
# 各bootstrapを使った訓練の実行時間を保存しておくための配列
executed_time = []
# 各bootstrap sampleで学習したモデルの test accuracyの配列
models_acc = []

print(f'dataset={dataset}, model_type={model_type}, device={params["device"]}')

# 各bootstrap sampleについて繰り返し
for i in range(args.start_boot_id, B):
  
  print("========= use bootstrap sample {} for training data =========".format(i))
  boot = load_pickle(get_path(os.path.join(getattr(DataPath, dataset.upper()).BOOT_DATA_DIR, "boot_{}.pkl".format(i)))) if not isTomita else \
    load_pickle(get_path(os.path.join(DataPath.TOMITA.BOOT_DATA_DIR.format(dataset), "boot_{}.pkl".format(i))))
  # train_x, train_y以外の属性は合わせる
  keys = list(data.keys()).copy()
  keys.remove('train_x'); keys.remove('train_y')
  for key in keys:
    boot[key] = data[key]
  # モデルの訓練を行う
  start_time = time.time()
  model, train_acc, test_acc = train(boot, params)
  fin_time = time.time()
  executed_time.append(fin_time - start_time)
  models_acc.append(test_acc)
  print("The training time for {} epochs: {:.2f}sec.".format(params["EPOCH"], executed_time[i-args.start_boot_id]))

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
print("The average accuracy of boot {}-{} models: {:.4f}".format(args.start_boot_id, B-1, sum(models_acc) / len(models_acc) ))
print("The total time of training for {} epochs in {}-{} bootstrap samples: {:.2f}sec.".format(params["EPOCH"], args.start_boot_id, B-1, sum(executed_time)))