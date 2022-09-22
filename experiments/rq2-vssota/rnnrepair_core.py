wimport os
import sys, glob
import argparse
# 異なる階層のutilsをインポートするために必要
sys.path.append("../")
sys.path.append("../../")
import numpy as np
import math
import time
import random
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve, roc_auc_score
from collections import defaultdict
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_pickle, load_json, save_json
# 異なる階層のmodel_utilsからインポート
from model_utils.model_util import load_model, sent2tensor
from model_utils import train_args
from rq2-vssota.reduction import PCA_R
from rq2-vssota.abstraction import GMM

def calculate_similarity_list(tr1, tr2_list, components, label=10):
  tr1_tensor = torch.from_numpy(tr1)
  traces = [torch.from_numpy(i) for i in tr2_list]
  traces.append(tr1_tensor)

  # pad_sequenceによって1次元目(文の長さ)を揃える，paddingする値はデフォルトで0
  total = torch.nn.utils.rnn.pad_sequence(traces, batch_first=True)

  total[:,:,1] = (total[:,:,1]+1) / (label+1)
  total[:,:,0] = (total[:,:,0]+1) /(components+1)

  tr1 = total[-1]
  tr2 = total[:-1]
  tr1 = torch.unsqueeze(tr1, dim=0) # shapeを(L, D) -> (1, L, D)にする
  tr1 = torch.cat([tr1]*len(tr2), dim=0) # tr2と同じ数だけtr2を並べる．これでtr1とtr2が同じ形状になる．tr1は同じ(L,D)のshapeのtensorがデータ数分並んだもので，tr2との比較ようにこうしている
  # print(tr1.shape)
  # print(tr2.shape)
  with torch.no_grad():
    # F.l1_loss(tr1, tr2, reduction="none") の shapeは(データ数, L, D)
    # これをmean([-1,-2])するので返り値のshapeは(データ数,)
    return F.l1_loss(tr1, tr2, reduction="none").mean([-1, -2]).numpy()

"""
やりたいこと
- 学習済みモデルでvalデータの隠れ状態のトレース取得
- 隠れ状態の次元をPCAで削減
- 削減した隠れ状態の集合からGMMを作る
"""

if __name__=='__main__':
  B = 10
  # コマンドライン引数から受け取り
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", type=str, help='abbrev. of datasets')
  parser.add_argument("model_type", type=str, help='type of models')
  parser.add_argument("theta", type=int, help='the ratio of extracted data from val data', choices=[10, 50])
  parser.add_argument("--device", type=str, help='cpu or id of gpu', default='cpu') # gpuならその番号(0,1,...)
  parser.add_argument("--start_boot_id", type=int, help='What boo_id starts from.', default=0)
  args = parser.parse_args()
  dataset, model_type, theta, device = args.dataset, args.model_type, args.theta, args.device
  start_boot_id = args.start_boot_id
  print("\ndataset = {}\nmodel_type = {}".format(dataset, model_type))

  # 目的変数のクラス数
  num_class = 10 if dataset == DataSet.MNIST else 2

  # bootごとのトレース取得時間を入れておく配列
  elapsed_abs_times = np.zeros((B))
  elapsed_sel_times = np.zeros((B))


  for i in range(start_boot_id, B):
    print("\n============== boot_id={} ==============".format(i))
    s_time = time.perf_counter()
    # con_trの保存ディレクトリ
    save_dir = get_path(f'./artifacts/{model_type}/{dataset}/boot_{i}')

    # val data
    val_trs = np.load(os.path.join(save_dir, 'con_tr_val.npz'), allow_pickle=True)
    con_traces, pred_seq_labels, pred_labels, val_labels, val_scores = val_trs['con_traces'], val_trs['pred_seq_labels'], val_trs['pred_labels'], val_trs['val_labels'], val_trs['scores']
    # 隠れ状態トレースにPCA適用して次元削減するフェーズ
    k = 10
    pca = PCA_R(k)
    print('create pca...')
    pca_data, min_val, max_val = pca.create_pca(con_traces)
    print('done.')
    # print(len(pca_data)) # valデータの長さ
    # print(pca_data[1].shape) # valデータの1番目のデータの各系列での隠れ状態を10次元にしたもの．つまりshapとしては (一番目のvalデータの長さ， k(=10))
    # ここまででRNNRepairのdatagen.pyの179行目までそろった

    # テストデータに対して同じPCAを適用して次元削減
    test_trs = np.load(os.path.join(save_dir, 'con_tr_test.npz'), allow_pickle=True)
    test_con_traces, test_pred_seq_labels, test_pred_labels, test_labels = test_trs['test_con_traces'], test_trs['test_pred_seq_labels'], test_trs['test_pred_labels'], test_trs['test_labels']
    print('apply pca...')
    test_pca_data = pca.do_reduction(test_con_traces)
    print('done')
    # print(len(test_pca_data), test_pca_data[0].shape)

    # testデータでミスしたデータのインデックスの配列
    is_correct_test = (np.array(test_pred_labels) == np.array(test_labels))
    test_missed_indices = np.where(is_correct_test==False)[0]

    # valデータに対するpca_dataからGMMのモデルを作る
    m = 7 # GMMの成分数
    print('create gmm...')
    ast_model = GMM([pca_data], m, pred_seq_labels, num_class)
    print('done.')
    f_time = time.perf_counter()
    elapsed_abs_times[i] = f_time - s_time
    print(f'epalsed for abstraction: {elapsed_abs_times[i]} [sec.]')

    s_time = time.perf_counter()
    # GMM上でトレースとる
    val_gmm_trace = ast_model.get_trace_input(pca_data, pred_seq_labels)
    test_gmm_trace = ast_model.get_trace_input(test_pca_data, test_pred_seq_labels)

    # ミスしたテストデータのトレースと，valの各データのトレースの類似度を計算
    # ミスした各データに対してvalの各データとのトレースの類似度計算
    # ->ミスした各データとの類似度を平均して上位10件を取り出す
    similarities_avg = np.zeros(val_labels.shape)
    for j, index in enumerate(test_missed_indices):
      # print(f'missed index: {index}')
      # L1lossを計算しているので値が小さい方が類似度が高い
      similarities_avg += calculate_similarity_list(test_gmm_trace[index], val_gmm_trace, m, label=num_class)
    similarities_avg /= len(test_missed_indices) # 平均するために割り算
    # ロスの値で昇順ソート（=類似度の高い順にソート）
    sort_indeces = np.argsort(similarities_avg)
    # あとはこのインデックスを使ってvalデータからデータを取り出す
    # 取り出すデータの件数
    num_extract = math.floor(len(sort_indeces)*theta/100)
    # データを抽出
    extract_indeces = sort_indeces[:num_extract]
    # 抽出したデータの予測ラベルと正しいラベルを取得
    y_pred, y_true = np.array(pred_labels)[extract_indeces], np.array(val_labels)[extract_indeces]
    # 抽出データの予測スコアを取得
    val_scores = val_scores[extract_indeces]

    # 抽出したデータのインデックス
    # 各抽出データに対する予測結果の平均を保存
    # aucだけめんどいので先に計算
    average_mode = 'binary' if num_class == 2 else 'macro'
    if num_class == 2:
      try:
        auc = roc_auc_score(y_true, val_scores)
      except ValueError as e:
        print(e)
        auc = np.nan
    else:
      try:
        auc = roc_auc_score(y_true, val_scores, average=average_mode, multi_class='ovo')
      except ValueError as e:
        print(e)
        auc = np.nan
    # 結果を保存
    save_path = os.path.join(save_dir, f'pred_res_theta={theta}')
    np.savez(save_path, extract_indeces=extract_indeces, \
      accuracy=accuracy_score(y_true, y_pred), precision=precision_score(y_true, y_pred, average=average_mode), \
      recall=recall_score(y_true, y_pred, average=average_mode), f1=f1_score(y_true, y_pred, average=average_mode), \
      mcc=matthews_corrcoef(y_true, y_pred), auc=auc)
    f_time = time.perf_counter()
    elapsed_sel_times[i] = f_time - s_time
    print(f'epalsed for selection: {elapsed_sel_times[i]} [sec.]')
    print(f'pred_res is saved in {save_path}')
  abs_avg_time = np.mean(elapsed_abs_times)
  sel_avg_time = np.mean(elapsed_sel_times)
  print(f'\nmean abstraction time in {B} boots: {abs_avg_time} [sec.]')
  print(f'mean selection (theta=10) time in {B} boots: {sel_avg_time} [sec.]')
  with open(f'artifacts/{model_type}/abstraction_time.csv', 'a') as f:
    f.write(f'{dataset}, {abs_avg_time}\n')
  with open(f'artifacts/{model_type}/selection_time.csv', 'a') as f:
    f.write(f'{dataset}, {sel_avg_time}\n')