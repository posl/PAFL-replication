import os
import sys
# 異なる階層のmodel_utils, utilsをインポートするために必要
sys.path.append("../../")
# 異なる階層のPAFL/extract_pfa/make_abstract_trace/make_abs_traceをインポートするために必要
sys.path.append("../")
import numpy as np
import torch
from collections import defaultdict
# 異なる階層utilsからのインポート
from utils.help_func import filter_stop_words, save_pickle
from utils.constant import *
# 異なる階層model_utilsからのインポート
from model_utils.model_util import sent2tensor
# 異なる階層make_abstract_traceからのインポート
from PAFL.extract_pfa.make_abstract_trace.make_abs_trace import level1_abstract
# 同じ階層get_pfa_spectrumからのインポート
from PAFL.get_pfa_spectrum.get_reachability_prob import get_state_reachability

def extract_l1_trace(x, model, input_dim, device, partitioner, word2idx=None, wv_matrix=None, pt_type=PartitionType.KM, use_clean=True, num_class=2):
  """
  サンプル x のabstract traceを取得する.
  abstract trace取得のために, 学習済みのpartitionerを利用する.
  
  Parameters
  ------------------
  x: list of str
    tensorに変換されるサンブル(文)
  model: torch.nn.Module
    学習済みのRNNモデル
  input_dim: int
    RNNへの入力次元数(埋め込みベクトルの次元数)
  word2idx: dict
    単語 -> 単語idへの対応表
  wv_matrix: list of list of float
    単語idから埋め込みベクトルへの対応表(埋め込み行列)
    実態は(語彙数, input_dim)と言う形状の2次元配列
  device: str
    CPUを使う場合, "cpu"とする
  partitioner: make_abstract_trace.partitioner.Partitioner
    学習済みのクラスタリング関数(partitioner)
  pt_type: str
    Kmeansを使う場合, "KM"
  use_clean: bool
    fileter_stop_wordsを行う場合True, そうでなければFalse.
  num_class: int
    目的変数のクラス数．
  
  Returns
  ------------------
  abs_seq: list of str
    サンプル文 x を model に入力した際の隠れ状態たちを partitioner でクラスタリングして得られる abstract trace.
  """
  if use_clean:
    x = filter_stop_words(x)
  if (word2idx is not None) and (wv_matrix is not None):
    # xを文からtensorに変換する
    x_tensor = sent2tensor(x, input_dim, word2idx, wv_matrix, device)
  else:
    # xを画像（多分）からtensorに変換する
    x_tensor = torch.unsqueeze(torch.tensor(x), 0) / 255 # (1, 28, 28)
  # xを入力した時のRNNの各時刻での隠れ状態とラベルのリストを取得
  hn_trace, label_trace = model.get_predict_trace(x_tensor)
  # 最終的な予測ラベルのみを取り出す
  rnn_pred = label_trace[-1]
  # 学習済みのpartitonerを用いて隠れ状態ベクトルをクラスタに割り当てて得られるabs_traceを返す
  abs_seq = level1_abstract(rnn=None, rnn_traces=[hn_trace], y_pre=[rnn_pred],
                              partitioner=partitioner, num_class=num_class,
                              partitioner_exists=True, partition_type=pt_type)[0]
  return abs_seq

def trace_abs_seq(abs_seq, trans_func, trans_wfunc, total_states):
  """
  abstract traceを1つ入力し, pfaの状態におけるトレースに変換する. 
  そのためにpfaの正例が必要.
  
  Parameters
  ------------------
  abs_seq: list of str
    あるサンプルのabstract trace(extract_l1_traceの返り値)
  trans_func: dict
    PFAの遷移関数
  trans_wfunc: dict
    PFAの遷移関数
  total_states: int
    PFAの状態数
  
  Returns
  ------------------
  L2_trace: list of str
    abs_seqをPFA上でトレースした状態のログ
  pass_mat: list of list of int
    pass_mat[i][j] = PFAの状態iから状態jへの遷移があった場合1, そうでない場合0となる行列
  """
  # print(abs_seq)
  # 通過した遷移の行列
  pass_mat = np.zeros((total_states,total_states), dtype=int)
  # 初期状態
  s = 1
  L2_trace = [s]
  # abs_seqの各シンボルを1つずつ読み込む
  for i in range(1, len(abs_seq)):
    if str(abs_seq[i]) not in trans_wfunc[s]:
      L2_trace.append('T')
      break
    else:
      new_s = trans_func[s][str(abs_seq[i])]
      L2_trace.append(new_s)
      # print(f'now:{s}, in:{str(abs_seq[i])} => {new_s},  {L2_trace}')
      pass_mat[s-1][new_s-1] = 1
      s = new_s
  return L2_trace, pass_mat

def test_acc_fdlt(**kwargs):
  """
  引数にセットしたテストデータをPFAに入力してトレースをとることで, PFAでテストデータの予測を行う.
  
  Parameters
  ------------------
  X: list of list of str
    テストデータに含まれるサンプルのリスト
  Y: list of int
    テストデータに含まれるラベルのリスト
  dfa: dict
    learning_pfaで得られたpfa
  tmp_prism_data: str
    reachability計算用のpmファイル群のディレクトリのパス. get_state_reachabilityで必要.
  input_type: str
    入力データの種類. テキストデータの場合"text"を指定.
  model: torch.nn.Module
    学習済みのRNNモデル. extract_l1_traceで必要. 
  partitioner: partitioner.Partitioner
    学習済みのクラスタリング関数. extract_l1_traceで必要. 
  word2idx: dict
    単語 -> 単語idの対応表. extract_l1_traceで必要. 
  wv_matrix: list of list of float
    埋め込み行列. extract_l1_traceで必要. 
  input_dim: int
    入力の次元. extract_l1_traceで必要. 
  device: str
    CPUを使う場合は "CPU" とする. extract_l1_traceで必要. 
  total_states: int
    PFAの状態数. trace_abs_seqで必要.
  save_path: str
    pfa specrum(行列)を保存するディレクトリのパス.
  num_class: int
    目的変数のクラス数

  Returns
  ------------------
  pred_info: dict
    pfaによる予測の情報などをまとめた辞書. 主に以下の属性からなる.
    acc: float
      正解率. 正解数/データ数.
    fdlt: float
      忠実度. RNNとPFAの予測が一致した数/データ数.
    unspecified: int
      遷移が不定になったデータ数.
    rnn_conti_table: list of list of int
      元のRNNによる予測の分割表. 2 * 2 の形状になる. 
      rnn_conti_table[i][j] = RNNの予測がiで, 実際のラベルがjだったサンプル数. 
    pfa_conti_table: list of list of int
      PFAによる予測の分割表. 2 * 2 の形状になる. 
      pfa_conti_table[i][j] = PFAの予測がiで, 実際のラベルがjだったサンプル数. 
  """

  # キーワード引数の受け取り
  X = kwargs["X"]
  Y = kwargs["Y"]
  dfa = kwargs["dfa"]
  trans_func, trans_wfunc = dict(dfa["trans_func"]), dict(dfa["trans_wfunc"])
  tmp_prism_data = kwargs["tmp_prism_data"]
  model = kwargs["model"]
  partitioner = kwargs["partitioner"]
  input_dim = kwargs["input_dim"]
  device = kwargs["device"]
  total_states = kwargs["total_states"]
  save_path = kwargs["save_path"]
  num_class = kwargs["num_class"]
  is_multiclass = True
  if kwargs["input_type"] == "text":
    word2idx = kwargs["word2idx"]
    wv_matrix = kwargs["wv_matrix"]
    is_multiclass = False
  val_pred_labels = np.array(kwargs["val_pred_labels"])
  use_clean = kwargs['use_clean']

  # pfaによる予測の結果をまとめたjsonオブジェクトを作成
  pred_info = {}
  pred_info["total_states"] = total_states          # pfaの状態数
  pred_info["pfa_acc"] = 0                          # PFAの正解率
  pred_info["rnn_acc"] = 0                          # 元のRNNの正解率
  pred_info["fdlt"] = 0                             # 忠実度
  pred_info["unspecified"] = 0                      # 遷移不定の数
  # pred_info["rnn_conti_table"] = [[0, 0], [0, 0]]   # rnnの予測の分割表
  # pred_info["pfa_conti_table"] = [[0, 0], [0, 0]]   # pfaの予測の分割表

  # 変数の定義
  pmc_cache = {}  # 最終的なPFAの状態 => 各予測ラベルへのreachability probのキャッシュ
  # 成功, 失敗時のpass matrix(これがpfa spectrum)
  succ_pass_matrix = np.zeros((total_states,total_states), dtype=int)
  fail_pass_matrix = np.zeros((total_states,total_states), dtype=int)
  
  # 開始状態 -> 各ラベルへの到達確率(1次元配列)への辞書
  reachable_dict = defaultdict(list)
  # 状態 -> last_inner がその状態になるようなデータの数への辞書
  lastinner_count = defaultdict(int)
  # 状態 -> 成功,失敗数のリストへの辞書
  lastinner_result = defaultdict(lambda: [0, 0])
  # トレースから得られるRNNの予測ラベル
  rnn_pred_labels = []

  # 成功/失敗した場合の各データの状態遷移のログファイルが，既に存在する場合は消しておく
  os.remove('{}/succ_trace.txt'.format(save_path)) if os.path.exists('{}/succ_trace.txt'.format(save_path)) else None
  os.remove('{}/fail_trace.txt'.format(save_path)) if os.path.exists('{}/fail_trace.txt'.format(save_path)) else None

  # テストデータの各サンプルについて繰り返す
  for x, y in zip(X, Y):
    # サンプルのabstract traceを取得
    if kwargs['input_type'] == 'text':
      abs_trace = extract_l1_trace(x, model, input_dim, device, partitioner, word2idx=word2idx, wv_matrix=wv_matrix, num_class=num_class, use_clean=use_clean)
    else:
      abs_trace = extract_l1_trace(x, model, input_dim, device, partitioner, num_class=num_class, use_clean=use_clean)
    # print(abs_trace)
    # continue
    # abs_traceの最後の文字からrnnの予測を特定
    if not is_multiclass:
      rnn_pred = 0 if abs_trace[-1] == 'N' else 1
    else:
      rnn_pred = int(abs_trace[-1].lstrip('L'))
    rnn_pred_labels.append(rnn_pred)

    # abs_traceをpfa上でトレース
    L2_trace, pass_mat = trace_abs_seq(abs_trace, trans_func, trans_wfunc, total_states)
    # 予測ラベルの状態の前の最後の状態を取り出す
    last_inner = L2_trace[-2]
    # last_innerからのrachabilityがキャッシュで計算済みであればそれを使う
    if last_inner in pmc_cache:
      probs = pmc_cache[last_inner]
    # キャッシュになければreachabilityを計算してキャッシュに入れる
    else:
      probs = get_state_reachability(tmp_prism_data, num_prop=num_class, start_s=last_inner, is_multiclass=is_multiclass)
      pmc_cache[last_inner] = probs
    # reachabilityの最大のラベルをPFAの予測ラベルとする
    pfa_pred = np.argmax(probs)

    # reachable_dictの更新
    reachable_dict[last_inner] = probs
    # lastinner_countの更新
    lastinner_count[last_inner] += 1
    
    # 分割表の更新
    # pred_info["rnn_conti_table"][rnn_pred][y] += 1
    # pred_info["pfa_conti_table"][pfa_pred][y] += 1

    # rnn_predが正解ラベルと一致した場合
    if rnn_pred == y:
      pred_info["rnn_acc"] += 1
    # pfa_predが正解ラベルと一致した場合
    if pfa_pred == y:
      pred_info["pfa_acc"] += 1
      lastinner_result[last_inner][0] += 1
      succ_pass_matrix += pass_mat
      # 成功時のトレースを更新
      with open('{}/succ_trace.txt'.format(save_path), 'a') as f:
        f.write((','.join([str(l) for l in L2_trace]))+'\n')
    elif pfa_pred != y:
      lastinner_result[last_inner][1] += 1
      fail_pass_matrix += pass_mat
      # 失敗時のトレースを更新
      with open('{}/fail_trace.txt'.format(save_path), 'a') as f:
        f.write((','.join([str(l) for l in L2_trace]))+'\n')
    # pfa_predがrnn_predと一致した場合
    if pfa_pred == rnn_pred:
      pred_info["fdlt"] += 1
    if L2_trace[-1] == "T":
      pred_info["unspecified"] += 1

  rnn_pred_labels = np.array(rnn_pred_labels)
  if sum(rnn_pred_labels != val_pred_labels) != 0:
    print('error')
    exit()
  pred_info["rnn_acc"] = pred_info["rnn_acc"] / len(Y)
  pred_info["pfa_acc"] = pred_info["pfa_acc"] / len(Y)
  pred_info["fdlt"] = pred_info["fdlt"] / len(Y)
  # 成功/失敗時の遷移の行列をpklで保存
  save_pickle(os.path.join(save_path, "succ_pass_mat.pkl"), succ_pass_matrix)
  save_pickle(os.path.join(save_path, "fail_pass_mat.pkl"), fail_pass_matrix)
  
  return pred_info

