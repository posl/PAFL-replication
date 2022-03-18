import os
import sys
import glob 
import torch
# 異なる階層のutils, model_utilsをインポートするために必要
sys.path.append("../../")
import numpy as np
import pandas as pd
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle
# 異なる階層のmodel_utilsからインポート
from model_utils.model_util import load_model, get_model_file
from model_utils import train_args
from model_utils.trainer import test4eval

if __name__ == "__main__":
  B = 10
  # モデルのタイプはLSTMで固定
  model_type = ModelType.LSTM
  # デバイスの設定
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  for i in range(B):
    print("========= use bootstrap sample {} for training data =========".format(i))

    # pred_res_dfの保存ディレクトリのパスを取得
    pred_res_save_dir = TrainedModel.PRED_RESULT_FOR_LOOK
    # pred_res_dfの保存ディレクトリが存在しなかったら, 作成する
    if not os.path.isdir(pred_res_save_dir):
      os.makedirs(pred_res_save_dir)

    # pred_res_dfを作成
    pred_res_df = pd.DataFrame(index=[], columns=[])
    # dfの列のリスト
    columns = ['dataset']
    
    for dataset in ['1', '2', '3', '4', '5', '6', '7', DataSet.BP, DataSet.MR, DataSet.IMDB]:
      isTomita = dataset.isdigit()
      # オリジナルのデータセットをロード
      data = load_pickle(get_path(getattr(DataPath, dataset.upper()).SPLIT_DATA)) if not isTomita  else \
        load_pickle(get_path(DataPath.TOMITA.SPLIT_DATA.format(dataset)))
      print("\ndataset = {}\nmodel_type= {}".format(dataset, model_type)) if not isTomita  else \
        print("\ndataset = {}\nmodel_type= {}".format("tomita_" + dataset, model_type))

      # モデルのパラメータを保持しておく
      params = getattr(train_args, "args_{}_{}".format(model_type, dataset))() if not isTomita  else \
        getattr(train_args, "args_{}_{}".format(model_type, "tomita"))()
      # 埋め込み行列のロード
      wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX)) if not isTomita  else \
        load_pickle(get_path(DataPath.TOMITA.WV_MATRIX.format(dataset)))
      params["WV_MATRIX"] = wv_matrix
      params["device"] = device
      params["use_clean"] = 0
      params["rnn_type"] = model_type
      pred_res_row = {}
      pred_res_row["dataset"] = dataset

      # モデルを読み出す
      model_dir = os.path.join(getattr(getattr(TrainedModel, model_type.upper()), dataset.upper()), "boot_{}".format(i)) if not isTomita  else \
        os.path.join(getattr(TrainedModel, model_type.upper()).TOMITA.format(dataset), "boot_{}".format(i))
      load_model_path = glob.glob(get_path(os.path.join(model_dir, "*.pkl")))[0]
      model = load_model(model_type, dataset, device, load_model_path)

      # テストデータの予測を行い，メトリクスを算出する
      print("measuring metrics...")
      pred_res = test4eval(data, model, params, device, "test_x", "test_y")
      # rocとconf_matは最終的なcsvに載せられないので除く
      pred_res.pop("roc")
      pred_res.pop("conf_mat")
      # 構成したpred_res_rowをdfに追加
      pred_res_df = pred_res_df.append(pred_res, ignore_index=True)
    
    # 列の順番を調整
    pred_res_df = pred_res_df.reindex(columns=["accuracy", "precision", "recall", "f1_score", "mcc", "auc"])
    # boot_idごとにpred_res_dfをcsvで保存する
    pred_res_df.to_csv(os.path.join(pred_res_save_dir, "boot_{}.csv".format(i)), header=True, index=False, sep=",")
    print("predict result is saved to ", os.path.join(pred_res_save_dir, "boot_{}.csv".format(i)))
    