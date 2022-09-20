import os
import sys
sys.path.append("../../") # extract_pfa/make_abstract_traceをインポートするために必要
sys.path.append("../../../") # utils, model_utilsをインポートするために必要
import torch
import numpy as np
# 異なる階層のutils, model_utilsからのインポート
from utils.constant import START_SYMBOL, PartitionType, get_path
from model_utils.model_util import sent2tensor
# 同じ階層のmake_abstract_traceからのインポート
from PAFL.extract_pfa.make_abstract_trace.partitioner import Partitioner, Kmeans, EHCluster

def get_term_symbol(y_pre, binary=True):
  """
  予測ラベルからシンボルを返す
  0->"N"(ネガティブ), 1->"P"(ポジティブ)
  
  Parameters
  ------------------
  y_pre: int
    予測ラベル. 0もしくは1.
  
  Returns
  ------------------
  : str
    予測ラベルに対応するシンボル(文字)
  """
  if binary:
    if y_pre == 0:
      return 'N'
    elif y_pre == 1:
      return "P"
  else:
    return f"L{y_pre}"

def _hn2probas(hn_vec, rnn):
  """
  隠れ状態ベクトルから確率のベクトルに変換する.
  隠れ状態でなく確率ベクトルでクラスタリングしたい場合に使う.
  (注)'_'からはじまってるモジュールはimport *でインポートされない.
  
  Parameters
  ------------------
  hn_vec: torch.Tensor
    隠れ状態ベクトルを表すtensor
  rnn: torch.nn.Module
    model_util/gated_rnnで定義されているLSTMもしくはGRU
  
  Returns
  ------------------
  probas: list of float
    各ラベルへの所属確率のベクトル
  """
  # 隠れ状態ベクトルの次元を増やす
  tensor = torch.unsqueeze(torch.tensor(hn_vec), 0) # (1, {hn_vecの形状})と言う形状のtensorになる
  # output_pr_ditrはsoftmaxの結果(各ラベルへ所属確率のベクトル)
  probas = rnn.output_pr_dstr(tensor).cpu().detach().squeeze().numpy() # numpy配列に変換
  return probas

def _rnn_trace2point_probas(rnn_traces, rnn):
  """
  original traceをもとに, 各データに対する, 各時刻での確率ベクトルのリストを出力する.
  よって, 出力のリスト(numpy配列)は2次元配列になる.
  
  Parameters
  ------------------
  rnn_traces: list of list of float
    RNNのoriginal trace
  rnn: torch.nn.Module
    model_util/gated_rnnで定義されているLSTMもしくはGRU

  Returns
  ------------------
  input_points: list of list of float
    各データに対する, 各時刻での確率ベクトル(softmaxの出力)
  seq_len: list of int
    各original traceの長さ
  """
  seq_len = []
  input_points = []
  # 各original traceに対するループ
  for seq in rnn_traces:
    seq_len.append(len(seq))
    # original trace内の各隠れ状態ベクトルに対するループ
    for hn_state in seq:
      # 隠れ状態ベクトルから確率ベクトル(softmaxの出力)を得る
      probas = _hn2probas(hn_state, rnn)
      input_points.append(probas)
  input_points = np.array(input_points)
  return input_points, seq_len

def _rnn_traces2point(rnn_traces):
  """
  original traceをもとに, 各データに対する, 各時刻での隠れ状態ベクトルのリストを出力する.
  よって, 出力のリスト(numpy配列)は2次元配列になる.
  _rnn_trace2point_probasとの違いは, input_pointsに格納するのが確率ベクトルか隠れ状態ベクトルかの違い.
  
  Parameters
  ------------------
  rnn_traces: list of list of float
    RNNのoriginal trace
  
  Returns
  ------------------
  input_points: list of list of float
    各データに対する, 各時刻での隠れ状態ベクトル
  seq_len: list of int
    各original traceの長さ
  """
  seq_len = []
  input_points = []
  # 各original traceに対するループ
  for seq in rnn_traces:
    seq_len.append(len(seq))
    # original trace内の各隠れ状態ベクトルに対するループ
    for hn_state in seq:
      # 隠れ状態ベクトルを配列input_pointsの要素として追加
      input_points.append(hn_state) # すべての隠れ状態ベクトルを1つの配列にまとめる
  input_points = np.array(input_points)
  return input_points, seq_len

def make_L1_abs_trace(labels, seq_len, y_pre, num_class):
  """
  隠れ状態ベクトル(or 確率ベクトル)をクラスタリングしたリストを, サンプルごとに1つにまとめる.
  出力のイメージとしては, 
  [
    [1つめのサンプルのabstract trace],
    [2つめのサンプルのabstract trace],
    ...
  ]
  って感じ.

  隠れ状態ベクトルのリスト: list of list of float
  -> クラスタリングされた隠れ状態ベクトルのリスト: list of str
  -> クラスタリングされた隠れ状態ベクトルのリストをサンプルごとにまとめたリスト: list of list of str

  Parameters
  ------------------
  labels: list of str
    すべての隠れ状態ベクトル(or 確率ベクトル)をクラスタリングしてラベルに割り当てた配列
  seq_len: list of int
    各サンプルの長さを保持する配列
  y_pre: list of int
    予測ラベルのリスト
  num_class: int
    目的変数のクラス数
  
  Returns
  ------------------
  abs_seqs: list of list of str
    隠れ状態ベクトル(or 確率ベクトル)をクラスタリングした配列を, サンプルごとにまとめた配列
  """
  start_p = 0
  abs_seqs = []
  # 目的変数がバイナリかどうかのフラグ
  binary = True if num_class == 2 else False
  for size, y in zip(seq_len, y_pre):
    # input_points の各データのラベルのリストから，各データのサイズの分だけスライスする
    abs_trace = labels[start_p:start_p + size]
    term_symbol = get_term_symbol(y, binary)
    # 各データに対する abstract trace をここで形成している
    abs_trace = [START_SYMBOL] + abs_trace + [term_symbol]
    abs_seqs.append(abs_trace)
    start_p += size
  return abs_seqs

def level1_abstract(**kwargs):
  """
  original traceに対してクラスタリングを行うことでabstract traceを構成する.
  得られたabstract traceと, クラスタリングに用いたpartitionerを返す.

  Parameters
  ------------------
  partitioner_exists: bool, required.
    whether or not to use an pre-trained partitioner
  rnn_traces:list(list), required.
    the execute trace of each text on RNN
  y_pre:list, required
    the label of each text given by RNN
  k: int, required when 'kmeans_exists' is false.
    number of clusters to form.
  partitioner: the object of sklearn.cluster.KMeans, required when 'partitioner_exists' is True.
    pre-trained kmeans.
  partition_type: str, option:[km|km-p|hc], required if partitioner_exists is false
  rnn: rnn model. instance of target_models.my_module.Mymodul.
  
  Returns
  ------------------
  abs_seqs: list(list).
    the level1 abstraction of each rnn trance
  kmeans: the object of sklearn.cluster.KMeans, returned onlt when 'kmeans_exists' is False.
  """

  rnn_traces = kwargs["rnn_traces"]
  y_pre = kwargs["y_pre"]
  pt_type = kwargs["partition_type"]
  num_class = kwargs["num_class"]

  # 隠れ状態ベクトルでなく, 確率ベクトルでクラスタリングする場合
  if pt_type == PartitionType.KMP:
    rnn = kwargs["rnn"]
    input_points, seq_len = _rnn_trace2point_probas(rnn_traces, rnn)
  # 隠れ状態ベクトルでクラスタリングする場合
  else:
    input_points, seq_len = _rnn_traces2point(rnn_traces)

  # 事前に別で用意した学習済みpartitionerを用いる場合
  if kwargs["partitioner_exists"]:
    partioner = kwargs["partitioner"]
    labels = list(partioner.predict(input_points))
    abs_seqs = make_L1_abs_trace(labels, seq_len, y_pre, num_class)
    return abs_seqs
  # このプログラム内でクラスタリングの学習から実行する場合
  else:
    # クラスタ数
    k = kwargs["k"]
    # 階層型クラスタリングの場合
    if pt_type == PartitionType.HC:
      partitioner = EHCluster(n_clusters=k)
      partitioner.fit(input_points)
    # KMeansクラスタリングの場合
    else:
      partitioner = Kmeans(k)
      partitioner.fit(input_points)

    # input_pointsの各データに対するラベルのリストを返す
    labels = partitioner.get_fit_labels()
    # abstract trace の形に整形する
    abs_seqs = make_L1_abs_trace(labels, seq_len, y_pre, num_class)
    return abs_seqs, partitioner

def save_level1_traces(abs_seqs, output_path):
  """
  abs_seqsを, output_pathで指定したパスに保存する. 
  
  Parameters
  ------------------
  abs_seqs: list of list of str
    lebel1_abstractで得られたabstract traces
  output_path: str
    abstract tracesの保存先のパス
  """
  output_path = get_path(output_path)
  # ディレクトリ/file.extを, os.path.splitにより, (ディレクトリ, file.ext)というタプルに変換する
  directory = os.path.split(output_path)[0]
  if not os.path.exists(directory):
    os.makedirs(directory)
  with open(output_path, "wt") as f:
    for seq in abs_seqs:
      line = ",".join([str(ele) for ele in seq])
      f.write(line + "\n")
  print("abstract traces saved to {}".format(output_path))
