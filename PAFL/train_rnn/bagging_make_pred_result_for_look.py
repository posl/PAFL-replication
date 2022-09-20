import os
import sys
import glob 
import torch
from collections import defaultdict
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
  # どのデータを評価対象とするか
  mode = "test"
  assert mode in ["test", "val"], "modeはtrainかtestにしてください．"
  # boot数
  B = 10
  # モデルのタイプはLSTMで固定
  model_type = ModelType.LSTM
  # デバイスの設定
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # pred_res_dfの保存ディレクトリのパスを取得
  pred_res_save_dir = getattr(TrainedModel, model_type.upper()).PRED_RESULT_FOR_LOOK
  # pred_res_dfの保存ディレクトリが存在しなかったら, 作成する
  if not os.path.isdir(pred_res_save_dir):
    os.makedirs(pred_res_save_dir)
  # pred_res_dfを作成
  pred_res_df = pd.DataFrame(index=[], columns=[])
  exit()
  
  for dataset in ['3', '4', '7', DataSet.BP, DataSet.MR, DataSet.IMDB, DataSet.MNIST, DataSet.TOXIC]:
    # modelを入れる配列
    models = [] * B
    # tomitaかどうかのフラグ
    isTomita = True if dataset.isdigit() else False
    # mnistかどうかのフラグ
    isImage = True if dataset == DataSet.MNIST else False
    
    # オリジナルのデータセットをロード
    data = load_pickle(get_path(getattr(DataPath, dataset.upper()).SPLIT_DATA)) if not isTomita  else \
      load_pickle(get_path(DataPath.TOMITA.SPLIT_DATA.format(dataset)))
    # 埋め込み行列のロード
    if not isImage:
      wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX)) if not isTomita  else \
        load_pickle(get_path(DataPath.TOMITA.WV_MATRIX.format(dataset)))
    print("\ndataset = {}\nmodel_type= {}".format(dataset, model_type)) if not isTomita  else \
      print("\ndataset = {}\nmodel_type= {}".format("tomita_" + dataset, model_type))

    # モデルのパラメータを保持しておく
    params = getattr(train_args, "args_{}_{}".format(model_type, dataset))() if not isTomita  else \
      getattr(train_args, "args_{}_{}".format(model_type, "tomita"))()
    params["rnn_type"] = model_type
    params["device"] = device
    if not isImage:
      params["WV_MATRIX"] = wv_matrix
      params["use_clean"] = 0

    pred_res = defaultdict(float)

    # 各bootのモデルをロード
    for i in range(B):        
      model_dir = os.path.join(getattr(getattr(TrainedModel, model_type.upper()), dataset.upper()), "boot_{}".format(i)) if not isTomita  else \
                  os.path.join(getattr(TrainedModel, model_type.upper()).TOMITA.format(dataset), "boot_{}".format(i))
      load_model_path = glob.glob(get_path(os.path.join(model_dir, "*.pkl")))[0]
      # モデルをロード
      model = load_model(model_type, dataset, device, load_model_path)
      # 予測を実行しメトリクスを得る
      tmp_dic = test4eval(data, model, params, key_x='test_x', key_y='test_y')
      # rocとconf_matは最終的なcsvに載せられないので除く
      tmp_dic.pop("roc")
      tmp_dic.pop("conf_mat")
      for m, v in tmp_dic.items():
        pred_res[m] += v
    for m in pred_res.keys():
      pred_res[m] *= 0.1

    # 構成したpred_res_rowをdfに追加
    pred_res_df = pred_res_df.append(pred_res, ignore_index=True)
    print(pred_res_df)
  
  # 列の順番を調整
  pred_res_df = pred_res_df.reindex(columns=["accuracy", "precision", "recall", "f1_score", "mcc", "auc"])
  # pred_res_dfをcsvで保存する
  pred_res_df.to_csv(os.path.join(pred_res_save_dir, "{}_bagging_result.csv".format(mode, i)), header=True, index=False, sep=",")
  print("predict result is saved to ", os.path.join(pred_res_save_dir, "{}_bagging_result.csv".format(mode,i)))
