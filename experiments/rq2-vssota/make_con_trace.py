import os
import sys, glob
import argparse
# 異なる階層のutilsをインポートするために必要
sys.path.append("../")
sys.path.append("../../")
import numpy as np
import time
import re
import random
import torch
from sklearn.metrics import accuracy_score
from collections import defaultdict
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_pickle, load_json, save_json
# 異なる階層のmodel_utilsからインポート
from model_utils.model_util import load_model, sent2tensor
from model_utils import train_args

if __name__=='__main__':
  B = 10
  # コマンドライン引数から受け取り
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", type=str, help='abbrev. of datasets')
  parser.add_argument("model_type", type=str, help='type of models')
  parser.add_argument("--device", type=str, help='cpu or id of gpu', default='cpu') # gpuならその番号(0,1,...)
  parser.add_argument("--start_boot_id", type=int, help='What boo_id starts from.', default=0)
  args = parser.parse_args()
  dataset, model_type, device= args.dataset, args.model_type, args.device
  start_boot_id = args.start_boot_id
  print("\ndataset = {}\nmodel_type = {}".format(dataset, model_type))
  
  # データセットに関する情報を設定
  isTomita = dataset.isdigit()
  isImage = (dataset == DataSet.MNIST)
  num_class = 10 if isImage else 2

  # 前処理済みのデータをロードしてくる
  ori_data = load_pickle(get_path(getattr(DataPath, dataset.upper()).SPLIT_DATA)) if not isTomita else \
    load_pickle(get_path(DataPath.TOMITA.SPLIT_DATA.format(dataset)))
  if dataset != DataSet.MNIST:
    word2idx = ori_data["word_to_idx"]
    # word2vecにかけた後のデータをロードしてくる
    wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX)) if not isTomita else \
      load_pickle(get_path(DataPath.TOMITA.WV_MATRIX.format(dataset)))

  # 入力する埋め込みベクトルの次元(データセットごとに異なる)
  if dataset == DataSet.BP:
    input_dim = 29
  elif dataset == DataSet.MR or dataset == DataSet.IMDB or dataset == DataSet.TOXIC:
    input_dim = 300
  elif dataset == DataSet.MNIST:
    input_dim = 28
  else: # tomita
    input_dim = 3
  
  # bootごとのトレース取得時間を入れておく配列
  elapsed_times = np.zeros((B))
  
  for i in range(start_boot_id, B):
    print("\n============== boot_id={} ==============".format(i))
    # モデルを読み出す
    model_dir = os.path.join(getattr(getattr(TrainedModel, model_type.upper()), dataset.upper()), "boot_{}".format(i)) if not isTomita  else \
      os.path.join(getattr(TrainedModel, model_type.upper()).TOMITA.format(dataset), "boot_{}".format(i))
    load_model_path = glob.glob(get_path(os.path.join(model_dir, "*.pkl")))[0]
    print("load_model...")
    model = load_model(model_type, dataset, device, load_model_path) if not isTomita else \
      load_model(model_type, dataset, device, load_model_path)
    print("done.")
    # モデルのパラメータを保持しておく
    params = getattr(train_args, "args_{}_{}".format(model_type, dataset))() if not isTomita else \
      getattr(train_args, "args_{}_{}".format(model_type, "tomita"))()
    params["device"] = device
    params["is_image"] = isImage
    params["is_multiclass"] = True if num_class > 2 else False
    if not isImage:
      params["WV_MATRIX"] = wv_matrix
    # print(model)
    # 出力の保存先
    save_path = get_path(f'./artifacts/{model_type}/{dataset}/boot_{i}')
    os.makedirs(save_path, exist_ok=True)

    for mode in ['val', 'test']:
      # modeがvalかtestかで分岐する
      if mode == 'val':
        print('mode: val')
        # boot{i}でのvalデータを読み出す
        val_boot_dir = getattr(DataPath, dataset.upper()).BOOT_DATA_DIR if not isTomita else \
          DataPath.TOMITA.BOOT_DATA_DIR.format(dataset)
        val_boot_path = os.path.join(val_boot_dir, f'val_boot_{i}.pkl')
        val_boot = load_pickle(get_path(val_boot_path))
        val_labels = val_boot['val_y']
        print(f'len of val data = {len(val_boot["val_x"])}')

        # 生の隠れ状態のトレースを取得するフェーズ
        con_traces = []
        pred_seq_labels = [] # リストの長さは#val_dataになる．各要素はval_dataのi番目のデータの系列長に等しいリストで，各時点での予測ラベルが入る
        pred_labels = [] # valの各データの予測ラベル
        scores = [] # acu計算のためのスコア
        s_time = time.perf_counter()
        for x, y in zip(val_boot["val_x"], val_boot["val_y"]):
          if dataset == DataSet.MNIST:
            tensor_sequence = torch.unsqueeze(torch.tensor(x), 0) / 255 # (1, 28, 28)
          else:
            tensor_sequence = sent2tensor(x, input_dim, word2idx, wv_matrix, device)
          tensor_sequence = tensor_sequence.to(device)
          hn_trace, label_trace = model.get_predict_trace(tensor_sequence)
          hn_trace, label_trace = np.array(hn_trace), np.array(label_trace)
          con_traces.append(hn_trace)
          pred_seq_labels.append(label_trace)
          pred_labels.append(label_trace[-1])
          # auc計算のためのスコアを取得しておく
          score = model.h2o(torch.from_numpy(hn_trace[-1]).float().unsqueeze(0)) # 最終時刻での隠れ状態をtorch.Tensorにしてunsqueezeで形状を(1,X)にする
          prob = torch.exp(model.softmax(score))
          prob = prob.cpu().data.numpy()
          # print(prob)
          if num_class == 2:
            scores.append(prob[0][1]) # 2値分類の場合はポジティブの確率
          else:
            scores.append(prob[0])    # 多値分類の場合は確率の配列
        con_traces, pred_seq_labels, pred_labels, scores = np.array(con_traces), np.array(pred_seq_labels), np.array(pred_labels), np.array(scores)
        np.savez(os.path.join(save_path, 'con_tr_val'), con_traces=con_traces, pred_seq_labels=pred_seq_labels, pred_labels=pred_labels, val_labels=val_labels, scores=scores)
        f_time = time.perf_counter()
        elapsed_times[i] += f_time - s_time
        print(f'con_tr_val is saved in {os.path.join(save_path, "con_tr_val")}')
        print(f'\nelapsed = {elapsed_times[i]} [sec.]')
    
      elif mode == 'test':
        print('mode: test')
        print(f'len of test data = {len(ori_data["test_x"])}')
        test_con_traces = []
        test_pca_data = []
        test_pred_seq_labels = [] # リストの長さは#test_dataになる．各要素はval_dataのi番目のデータの系列長に等しいリストで，各時点での予測ラベルが入る
        test_pred_labels = [] # testの各データの予測ラベル
        test_labels = ori_data['test_y']
        s_time = time.perf_counter()
        for x, y in zip(ori_data["test_x"], ori_data["test_y"]):
          if dataset == DataSet.MNIST:
            tensor_sequence = torch.unsqueeze(torch.tensor(x), 0) / 255 # (1, 28, 28)
          else:
            tensor_sequence = sent2tensor(x, input_dim, word2idx, wv_matrix, device)
          tensor_sequence = tensor_sequence.to(device)
          hn_trace, label_trace = model.get_predict_trace(tensor_sequence)
          hn_trace, label_trace = np.array(hn_trace), np.array(label_trace)
          test_con_traces.append(hn_trace)
          test_pred_seq_labels.append(label_trace)
          test_pred_labels.append(label_trace[-1])
        test_con_traces, test_pred_seq_labels, test_pred_labels = np.array(test_con_traces), np.array(test_pred_seq_labels), np.array(test_pred_labels)
        np.savez(os.path.join(save_path, 'con_tr_test'), test_con_traces=test_con_traces, test_pred_seq_labels=test_pred_seq_labels, test_pred_labels=test_pred_labels, test_labels=test_labels)
        f_time = time.perf_counter()
        elapsed_times[i] += f_time - s_time
        print(f'con_tr_test is saved in {os.path.join(save_path, "con_tr_test")}')
        print(f'\nelapsed_collect_traces: {elapsed_times[i]} [sec.]')
  elapsed_avg = np.mean(elapsed_times)
  print(f'\nmean time in {B} boots: {elapsed_avg} [sec.]')
  with open(f'artifacts/{model_type}/collection_time.csv', 'a') as f:
    f.write(f'{dataset}, {elapsed_avg}\n')