import os
import sys 
import csv
import numpy as np
import math
import random
import argparse
from collections import defaultdict
import glob 
import torch
import matplotlib.pyplot as plt
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from utils.constant import START_SYMBOL
from utils.time_util import current_timestamp
from utils.help_func import filter_stop_words, load_pickle, save_pickle, save_json
from utils.constant import *
from model_utils.model_util import load_model
from DFA.dfa import *
from PAFL.extract_pfa.learning_pfa.read_abs_data import load_trace_data

def is_use_clean(dataset):
  if dataset == DataSet.MR or dataset == DataSet.IMDB or dataset == DataSet.TOXIC:
    use_clean = 1
  else:
    use_clean = 0
  return use_clean

def get_input_dim(dataset, isTomita):
  if isTomita:
    input_dim = 3
  elif isImage:
    input_dim = 28
  elif dataset == DataSet.BP:
    input_dim = 29
  else: # dataset == MR or IMDB
    input_dim = 300
  return input_dim

def prepare_input(dataset, model_type, data_source, i, k, total_symbols=1000000):
  ############################
  # load model and sentences
  ############################
  isTomita = dataset.isdigit()
  # モデルを読み出す
  model_dir = os.path.join(getattr(getattr(TrainedModel, model_type.upper()), dataset.upper()), "boot_{}".format(i)) if not isTomita  else \
    os.path.join(getattr(TrainedModel, model_type.upper()).TOMITA.format(dataset), "boot_{}".format(i))
  load_model_path = glob.glob(get_path(os.path.join(model_dir, "*.pkl")))[0]
  model = load_model(model_type, dataset, 'cpu', load_model_path)
  model.eval()
  # データを読み出す
  # オリジナルのデータの読み込み
  ori_data = load_pickle(get_path(getattr(DataPath, dataset.upper()).SPLIT_DATA)) if not isTomita else \
    load_pickle(get_path(DataPath.TOMITA.SPLIT_DATA.format(dataset)))
  # 埋め込み行列を読み出す
  wv_matrix = None
  if dataset != 'mnist':
    wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX)) if not isTomita else \
      load_pickle(get_path(getattr(DataPath, "TOMITA").WV_MATRIX).format(dataset))
  # bootstrap dataを読み出す
  data = load_pickle(get_path(os.path.join(getattr(DataPath, dataset.upper()).BOOT_DATA_DIR, f"boot_{i}.pkl"))) if not isTomita else \
    load_pickle(get_path(os.path.join(DataPath.TOMITA.BOOT_DATA_DIR.format(dataset), f"boot_{i}.pkl")))
  # ori_dataとtrain以外の属性を合わせる
  keys = list(ori_data.keys()).copy()
  keys.remove('train_x'); keys.remove('train_y')
  keys.remove('val_x'); keys.remove('val_y')
  for key in keys:
    data[key] = ori_data[key]
  # L1関連を読み出す
  pt_path = AbstractData.L1.format(model_type, dataset, i, k, "train_partition.pkl") if not isTomita else \
    AbstractData.L1.format(model_type, 'tomita_'+dataset, i, k, "train_partition.pkl")
  partitioner = load_pickle(pt_path)
  l1_trace_path = AbstractData.L1.format(model_type, dataset, i, k, "train.txt") if not isTomita else \
    AbstractData.L1.format(model_type, 'tomita_'+dataset, i, k, "train.txt")
  l1_traces, alphabet = load_trace_data(l1_trace_path, total_symbols)
  #################################
  # load L1 traces and partitioner
  ##################################
  return l1_traces, wv_matrix, data, model, partitioner

def get_label(symbol, num_class):
  if num_class == 2:
    return  0 if symbol=='N' else 1
  if num_class == 10:
    return int(symbol.lstrip('L'))


if __name__ == '__main__':
  # コマンドライン引数から受け取り
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", type=str, help='abbrev. of datasets')
  parser.add_argument("model_type", type=str, help='type of models')
  args = parser.parse_args()
  dataset, model_type = args.dataset, args.model_type
  print(f'----------dataset={dataset}, variant={model_type}----------')
  
  # もろもろ設定
  B = 10
  ks = range(2, 12, 2)
  device = "cpu"
  data_source = "train"
  isTomita = dataset.isdigit()
  isImage = (dataset == DataSet.MNIST)
  input_type = 'text' if not isImage else 'image'
  num_class = 10 if isImage else 2
  use_clean = is_use_clean(dataset)
  input_dim = get_input_dim(dataset, isTomita)

  for i in range(B):
    for k in ks:
      print(f'----------i={i}, k={k}----------')
      # 諸々の保存dir
      save_dir = get_path(f'data/dfa/{model_type}/{dataset}/boot_{i}/k={k}') if not isTomita else \
        get_path(f'data/dfa/{model_type}/tomita_{dataset}/boot_{i}/k={k}')
      os.makedirs(os.path.dirname(save_dir), exist_ok=True)
      # 必要なデータの保存
      l1_traces, wv_matrix, data, model, partitioner = prepare_input(dataset, model_type, data_source, i, k)
      # classifer = Classifier(model, _model_type, input_dim, data["word_to_idx"], wv_matrix, device)
      # ###########
      # # build FSA
      # ###########
      fsa = FSA(l1_traces, data["train_x"], k, data["vocab"], model, partitioner, use_clean)
      # fsaモデルのオブジェクトをpklで保存
      save_pickle(os.path.join(save_dir, 'fsa_model.pkl'), fsa)
      print(f'fsa model is saved in {os.path.join(save_dir, "fsa_model.pkl")}')

      # #############
      # # test FSA
      # #############
      fsa_acc, rnn_acc, fdlt = 0, 0, 0
      unspecified_cnt = 0
      # fsa情報のjson用の辞書
      fsa_info = {}
      # fsa上のトレース（全データ）用のlist
      fsa_traces = []
      # 成功/失敗時のトレース用のlist
      succ_traces, fail_traces = [], []
      # val_bootを読み出す
      val_data = load_pickle(get_path(os.path.join(getattr(DataPath, dataset.upper()).BOOT_DATA_DIR, f"val_boot_{i}.pkl"))) if not isTomita else \
        load_pickle(get_path(os.path.join(DataPath.TOMITA.BOOT_DATA_DIR.format(dataset), f"val_boot_{i}.pkl")))
      # val_l1traceを読み出す
      val_l1_trace_path = AbstractData.L1.format(model_type, dataset, i, k, "val_by_train_partition.txt") if not isTomita else \
        AbstractData.L1.format(model_type, 'tomita_'+dataset, i, k, "val_by_train_partition.txt")
      val_l1_traces, alphabet = load_trace_data(val_l1_trace_path, symbols_count=1000000)
      # 各データに対してfsaで予測してrnnの予測と比較する
      test_size = len(val_data["val_y"])
      for j, (x, y) in enumerate(zip(val_data["val_x"], val_data["val_y"])):
        if use_clean:
          x = filter_stop_words(x)
        pred, is_unspecified, fsa_trace = fsa.predict(x)
        fsa_traces.append(fsa_trace)
        # print(fsa_trace)
        rnn_pred = get_label(val_l1_traces[j][-1], num_class)
        # fsaトレースが正解した
        if pred == y:
          fsa_acc += 1
          succ_traces.append(fsa_trace)
        else:
          fail_traces.append(fsa_trace)
        if rnn_pred == y:
          rnn_acc += 1
        if pred == rnn_pred:
          fdlt += 1
        if is_unspecified:
          unspecified_cnt += 1
      fsa_acc = fsa_acc / test_size
      rnn_acc = rnn_acc / test_size
      fdlt = fdlt / test_size
      fsa_info['fdlt'], fsa_info['rnn_acc'], fsa_info['fsa_acc'], fsa_info['unspecified']  = fdlt, rnn_acc, fsa_acc, unspecified_cnt
      print(
          "k={}\tfsa_acc={:.4f}\trnn_acc={:.4f}\tfdlt={:.4f}\tfinal_states:{}\tunspecified_cnt:{}".format(k, fsa_acc, rnn_acc, fdlt, fsa.final_state, unspecified_cnt))
      # fsa情報をjsonで保存
      save_json(os.path.join(save_dir, 'fsa_info.json'), fsa_info )
      # fsa_traces, succ_traces, fail_tracesをtxtで保存
      with open(f'{save_dir}/val_trace.txt', 'w') as f:
        for trace in fsa_traces:
          f.write((','.join([str(l) for l in trace]))+'\n')
      with open(f'{save_dir}/succ_trace.txt', 'w') as f:
        for trace in succ_traces:
          f.write((','.join([str(l) for l in trace]))+'\n')
      with open(f'{save_dir}/fail_trace.txt', 'w') as f:
        for trace in fail_traces:
          f.write((','.join([str(l) for l in trace]))+'\n')
      
      
      """保存したいこと
      その次のプログラムで上のdictからval traceの各行に対してANS scoreを計算してデータを抽出する
      これで完璧！
      """