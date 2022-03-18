import os
import sys
import glob
import torch
# 異なる階層のutils, model_utilsをインポートするために必要
sys.path.append("../../")
import numpy as np
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_pickle
# 異なる階層のmodel_utilsからインポート
from model_utils.model_util import load_model, get_model_file
from model_utils import train_args
from model_utils.trainer import test4eval

if __name__ == "__main__":
  B = 10
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  # モデルのタイプはlstmで固定
  model_type = ModelType.LSTM
  # データセットはコマンドライン引数で受け取る
  dataset = sys.argv[1]
  isTomita = dataset.isdigit()
  # 変数datasetを変更する(tomitaの場合，"1" => "tomita_1"にする)
  if isTomita:
    tomita_id = dataset
    dataset = "tomita_" + tomita_id
  print("\ndataset = {}\nmodel_type= {}".format(dataset, model_type))
  # 埋め込み行列のロード
  wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX)) if not isTomita  else \
    load_pickle(get_path(DataPath.TOMITA.WV_MATRIX.format(tomita_id)))

  for i in range(B):
    print("\n============== boot_id={} ==============".format(i))

    # モデルを読み出す
    model_dir = os.path.join(getattr(getattr(TrainedModel, model_type.upper()), dataset.upper()), "boot_{}".format(i)) if not isTomita  else \
      os.path.join(getattr(TrainedModel, model_type.upper()).TOMITA.format(tomita_id), "boot_{}".format(i))
    load_model_path = glob.glob(get_path(os.path.join(model_dir, "*.pkl")))[0]
    model = load_model(model_type, dataset, device, load_model_path) if not isTomita else \
      load_model(model_type, tomita_id, device, load_model_path)
    # モデルのパラメータを保持しておく
    params = getattr(train_args, "args_{}_{}".format(model_type, dataset))() if not isTomita else \
      getattr(train_args, "args_{}_{}".format(model_type, "tomita"))()
    params["WV_MATRIX"] = wv_matrix
    params["device"] = device

    for n in range(1, 6):
      print("============== n={} ==============".format(n))

      for k in range(2, 12, 2):
        print("============== k={} ==============".format(k))

        for theta in [10, 50]:
          print("============== theta={} ==============".format(theta))

          for mode in ["ex", "rand"]:
            print("============== mode={} ==============".format(mode))

            # 抽出データorランダムデータのロード
            data_path = getattr(ExtractData, mode.upper()).format(model_type, dataset, i, k, n, theta)
            data = load_pickle(data_path)
            # dataをmodelで予測し, 様々なメトリクスを測定する
            print("make prediction for {} data...".format(mode))
            result = test4eval(data, model, params, device)
            print("done.")
            print("conf_mat:\n", result["conf_mat"])
            # 抽出データと同じディレクトリに, result(dict型)をjsonとして保存する
            save_pickle(os.path.join(ExtractData.DIR.format(model_type, dataset, i, k, n, theta), "{}_pred_result.pkl".format(mode)), result)