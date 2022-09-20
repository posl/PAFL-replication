"""
疑惑スコアの計算用の関数．
計算するために必要な疑惑n-gramや疑惑状態のロード関数もある．
"""

import os
import re
from collections import defaultdict
from utils.constant import *
from utils.help_func import load_pickle, load_json
# ゼロ除算を防ぐための定数
EPS = 1e-09

def get_susp_state_dict(model_type, dataset, boot_id, k, method="ochiai", target_source="val"):
  """
  入力情報から疑惑状態のpklファイルを特定しロードする

  Parameters
  ------------------
  model_type: str
    RNNモデルのタイプ.
  dataset: str
    データセットのタイプ.
  boot_id: int
    何番目のbootstrap samplingを使うか．
  k: int
    pfaのクラスタ数.
  method: str
    ochiai, dstar, tarantulaのいずれか.
  target_source: str
    疑惑値の抽出対象となるデータ(test, valなど)

  Returns
  ------------------
  susp_state_dict: dict
    状態の疑惑値dict(疑惑値の降順になっている)
  """
  # 疑惑状態の保存ファイルのパスを指定
  susp_state_dir = PfaSusp.STATE_SUSP.format(model_type, dataset, boot_id, k, target_source)
  susp_state_path = os.path.join(susp_state_dir, "{}_susp_dict.pkl".format(method))
  # 疑惑状態のpklファイルをロード
  susp_state_dict = load_pickle(susp_state_path)
  # dictのkeyだけint=>str型にして辞書を再構成する
  susp_state_dict = dict(list(
      map(lambda x : (str(x[0]),x[1]), susp_state_dict.items())
  ))
  return susp_state_dict

def get_susp_state(model_type, dataset, boot_id, k, method="ochiai", target_source="val"):
  """
  入力情報から疑惑状態のjsonファイルを特定しロードし, 疑惑値の最も高い状態を返す.
  jsonファイルには疑惑値の高い順に要素(状態, 疑惑値)が入っているため先頭要素のキーだけ返せば良い.

  Parameters
  ------------------
  model_type: str
    RNNモデルのタイプ.
  dataset: str
    データセットのタイプ.
  boot_id: int
    何番目のbootstrap samplingを使うか．
  k: int
    pfaのクラスタ数.
  method: str
    ochiai, dstar, tarantulaのいずれか.
  target_source: str
    疑惑値の抽出対象となるデータ(test, valなど)

  Returns
  ------------------
  s: str
    疑惑値の最も高い状態の番号
  """
  # 疑惑状態のpklファイルをロード
  susp_state_dict = get_susp_state_dict(model_type, dataset, boot_id, k, method, target_source)
  # 疑惑値の最も高い状態のみをリターン
  s = list(map(lambda x: x[0], susp_state_dict))[0]
  return s

def get_susp_ngram_dict(model_type, dataset, boot_id, k, ngram_n, method, target_source="val"):
  """
  入力情報から疑惑n-gramのpklファイルを特定しロードする

  Parameters
  ------------------
  model_type: str
    RNNモデルのタイプ.
  dataset: str
    データセットのタイプ.
  boot_id: int
    何番目のbootstrap samplingを使うか．
  k: int
    pfaのクラスタ数.
  ngram_n: int
    n-gramのn
  method: str
    ochiai, dstar, tarantulaのいずれか.
  target_source: str
    疑惑値の抽出対象となるデータ(test, valなど)

  Returns
  ------------------
  susp_ngram_dict: dict
    ngramの疑惑値のdict(疑惑値の降順になっている)
  """
  # 疑惑n-gramの保存ファイルのパスを指定
  susp_ngram_dir = PfaSusp.NGRAM_SUSP.format(model_type, dataset, boot_id, k, ngram_n, target_source)
  susp_ngram_path = os.path.join(susp_ngram_dir, "{}_susp_dict.pkl".format(method))
  # 疑惑n-gramのpklファイルをロード
  susp_ngram_dict = load_pickle(susp_ngram_path)
  # dictのkeyだけtuple=>str型にして辞書を再構成する
  susp_ngram_dict = dict(list(
      map(lambda x : (str(x[0]),x[1]), susp_ngram_dict.items())
  ))
  return susp_ngram_dict

def get_susp_ngram(model_type, dataset, boot_id, k, ngram_n, method="ochiai", target_source="val"):
  """
  入力情報から疑惑n-gramのjsonファイルを特定しロードし, 疑惑値の最も高いn-gramを返す.
  jsonファイルには疑惑値の高い順に要素(n-gram, 疑惑値)が入っているため先頭要素のキーだけ返せば良い.

  Parameters
  ------------------
  model_type: str
    RNNモデルのタイプ.
  dataset: str
    データセットのタイプ.
  boot_id: int
    何番目のbootstrap samplingを使うか．
  k: int
    pfaのクラスタ数.
  ngram_n: int
    n-gramのn
  method: str
    ochiai, dstar, tarantulaのいずれか.
  target_source: str
    疑惑値の抽出対象となるデータ(test, valなど)

  Returns
  ------------------
  susp_ngram: list
    疑惑値の最も高いngram. サイズはngram_nと一致する.
  """
  # 疑惑n-gramのpklファイルをロード
  susp_ngram_dict = get_susp_ngram_dict(model_type, dataset, boot_id, k, ngram_n, method, target_source)
  susp_ngram = list(susp_ngram_dict)[0]
  # 正規表現を使って数字の文字のリストにする(サイズはngram_n)
  susp_ngram = re.findall(r'\d+', susp_ngram)
  return susp_ngram

def count_susp_state_pass(susp_state, pfa_trace_path):
  """
  与えられたパスにあるpfa trace(.txt)をロードし, 各トレースにおける疑惑状態の通過回数を数える.
  i行目のトレースの疑惑状態の通過回数はsusp_state_pass[i]に格納する.
  
  Parameters
  ------------------
  susp_state: str
    疑惑状態を表す番号.
  pfa_trace_path: pfa tracesの格納パス.
  
  Returns
  ------------------
  susp_state_pass: defaultdict(int)
    トレースの行番号 -> 疑惑状態の通過回数 の対応辞書.
  """
  # pfa tracesを丸ごとロード
  with open(pfa_trace_path, "r") as f:
    pfa_traces = f.readlines()
  
  susp_state_pass = defaultdict(int)
  
  # 各行に対し, 疑惑状態の通過回数を数える
  for i, trace in enumerate(pfa_traces):
    trace = trace.strip().split(',')
    susp_state_pass[i] = 0
    for state in trace:
      if state == susp_state:
        susp_state_pass[i] += 1
  return susp_state_pass

def count_susp_ngram_pass(susp_ngram, pfa_trace_path):
  """
  与えられたパスにあるpfa trace(.txt)をロードし, 各トレースにおける疑惑ngramの通過回数を数える.
  i行目のトレースの疑惑ngramの通過回数はsusp_ngram_pass[i]に格納する.
  
  Parameters
  ------------------
  susp_ngram: list
    疑惑ngramを表す番号のリスト．
  pfa_trace_path: pfa tracesの格納パス.
  
  Returns
  ------------------
  susp_ngram_pass: defaultdict(int)
    トレースの行番号 -> 疑惑ngramの通過回数 の対応辞書.
  """
  ngram_n = len(susp_ngram)
  # pfa tracesを丸ごとロード
  with open(pfa_trace_path, "r") as f:
    pfa_traces = f.readlines()
  
  # 行(=データサンプル)ごとのsusp_gramの通過回数のカウンタ
  susp_ngram_pass = defaultdict(int)

  # 各行に対し, 疑惑ngramの通過回数を数える
  # pfa_traceの各行に対するループ
  for i, trace in enumerate(pfa_traces):
    trace = trace.strip().split(',')
    susp_ngram_pass[i] = 0

    # 行ごとの各状態に関するループ
    for j, state in enumerate(trace):
      # j+ngram_n > len(trace) の場合 ngram_n個のかたまりが作れないのでpass
      if (j+ngram_n) <= len(trace):
        ngram = trace[j:j+ngram_n]
        # 疑惑ngramと一致すればカウントアップ
        if ngram == susp_ngram:
          susp_ngram_pass[i] += 1
  return susp_ngram_pass

def score_state_rank_based(susp_state_dict, pfa_trace_path):
  """
  疑惑状態のランキングから，抽出の参考となるスコアを計算する
  
  Parameters
  ------------------
  susp_state_dict: dict
    状態の疑惑値が降順で並んだdict. キーが状態番号(str)で値が疑惑値(float).
  pfa_trace_path: str
    pfa tracesの格納パス.
  
  Returns
  ------------------
  state_rank_based_score: defaultdict(float)
    トレースの行番号 => rank based score への対応辞書．
  """
  # 状態 => 疑惑値ランキング の対応辞書を作る
  state_rank_dict = defaultdict(float)
  state_rank_dict = { state : float(r+1) for r, state in enumerate(susp_state_dict.keys()) }

  # pfa tracesを丸ごとロード
  with open(pfa_trace_path, "r") as f:
    pfa_traces = f.readlines()

  state_rank_based_score = defaultdict(float)
  # 各行に対し, rank based scoreをつけていく
  for i, trace in enumerate(pfa_traces):
    trace = trace.strip().split(',')
    state_rank_based_score[i] = 0.0
    for state in trace:
      if state != 'T':
        state_rank_based_score[i] += 1/state_rank_dict[state]
    # [0,1]の値にするために正規化
    state_rank_based_score[i] /= len(trace)
  return state_rank_based_score

def score_ngram_rank_based(susp_ngram_dict, n, pfa_trace_path):
  """
  疑惑ngramのランキングから，抽出の参考となるスコアを計算する
  
  Parameters
  ------------------
  susp_ngram_dict: dict
    ngramの疑惑値が降順で並んだdict. キーが状態番号のn項対(str)で値が疑惑値(float).
  n: int
    ngramのn.
  pfa_trace_path: str
    pfa tracesの格納パス.
  
  Returns
  ------------------
  ngram_rank_based_score: defaultdict(float)
    トレースの行番号 => rank based score への対応辞書．
  """
  # ngram => 疑惑値ランキング の対応辞書を作る
  ngram_rank_dict = defaultdict(float)
  ngram_rank_dict = { ngram : float(r+1) for r, ngram in enumerate(susp_ngram_dict.keys()) }

  # pfa tracesを丸ごとロード
  with open(pfa_trace_path, "r") as f:
    pfa_traces = f.readlines()

  ngram_rank_based_score = defaultdict(float)
  # 各行に対し, rank based scoreをつけていく
  for i, trace in enumerate(pfa_traces):
    trace = trace.strip().split(',')
    l = len(trace)
    ngram_rank_based_score[i] = 0.0
    ngram_l = 0
    for j, state in enumerate(trace):
      # j+n > l の場合 n個のかたまりが作れないのでpass
      if (j+n) <= l:
        ngram = trace[j:j+n]
        ngram = str(tuple(ngram))
        ngram_rank_based_score[i] += 1/ngram_rank_dict[ngram]
        ngram_l += 1
    # ngram1つあたりの平均にする
    if ngram_l != 0: 
      ngram_rank_based_score[i] /= ngram_l
  return ngram_rank_based_score

def score_relative_state_susp(susp_state_dict, pfa_trace_path):
  """
  状態の疑惑値の相対的な値から，抽出の参考となるスコアを計算する
  
  Parameters
  ------------------
  susp_state_dict: 状態の疑惑値が降順で並んだdict. キーが状態番号(str)で値が疑惑値(float).
  pfa_trace_path: pfa tracesの格納パス.
  
  Returns
  ------------------
  relative_state_susp_score: defaultdict(float)
    トレースの行番号 => relative state susp score への対応辞書．
  """
  # 疑惑値のトップ1の値を取得
  max_susp = max(susp_state_dict.values())
  
  # pfa tracesを丸ごとロード
  with open(pfa_trace_path, "r") as f:
    pfa_traces = f.readlines()

  relative_state_susp_score = defaultdict(float)
  # 各行に対し, relative_state_susp_scoreをつけていく
  for i, trace in enumerate(pfa_traces):
    trace = trace.strip().split(',')
    relative_state_susp_score[i] = 0.0
    for state in trace:
      if state != 'T':
        relative_state_susp_score[i] += susp_state_dict[state]
    relative_state_susp_score[i] /= max_susp
    relative_state_susp_score[i] /= len(trace)
  return relative_state_susp_score

def score_relative_ngram_susp(susp_ngram_dict, n, pfa_trace_path):
  """
  ngramの疑惑値の相対的な値から，抽出の参考となるスコアを計算する
  
  Parameters
  ------------------
  susp_ngram_dict: dict
    ngramの疑惑値が降順で並んだdict. キーが状態番号のn項対(str)で値が疑惑値(float).
  n: int
    ngramのn.
  pfa_trace_path: str
    pfa tracesの格納パス.
  
  Returns
  ------------------
  relative_ngram_susp_score: defaultdict(float)
    トレースの行番号 => relative ngram susp score への対応辞書．
  """
  # 疑惑値のトップ1の値を取得
  max_susp = max(susp_ngram_dict.values())

  # pfa tracesを丸ごとロード
  with open(pfa_trace_path, "r") as f:
    pfa_traces = f.readlines()

  relative_ngram_susp_score = defaultdict(float)
  # 各行に対し, relative_ngram_susp_scoreをつけていく
  for i, trace in enumerate(pfa_traces):
    trace = trace.strip().split(',')
    l = len(trace)
    relative_ngram_susp_score[i] = 0.0
    ngram_l = 0
    for j, state in enumerate(trace):
      # j+n > l の場合 n個のかたまりが作れないのでpass
      if (j+n) <= l:
        ngram = trace[j:j+n]
        ngram = str(tuple(ngram))
        relative_ngram_susp_score[i] += susp_ngram_dict[ngram]
        ngram_l += 1
    relative_ngram_susp_score[i] /= (max_susp + EPS) # 疑惑値が全部0の時の挙動がおかしくなるので + EPS しておく
    if ngram_l != 0: 
      relative_ngram_susp_score[i] /= ngram_l
  return relative_ngram_susp_score

def inverse_lookup(cnt_dict):  
  """
  行番号 => 通過回数 の辞書から，通過回数k => 通過回数==kである行数 の逆引き辞書を作る(ヒストグラム描くのに必要)
  Parameters
  ------------------
  cnt_dict: dict
    順方向の辞書

  Returns
  ------------------
  inv_dict: dict
    逆引き辞書
  """
  inv_dict = defaultdict(int)
  for key, cnt in cnt_dict.items():
    inv_dict[cnt] += 1
  return inv_dict
