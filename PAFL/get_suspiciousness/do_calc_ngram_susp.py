import os
import sys
import numpy as np
from pprint import pprint
from collections import defaultdict
#同じ階層のディレクトリからインポートするために必要
sys.path.append("../")
# 異なる階層のutilsをインポートするために必要
sys.path.append("../../")
# 異なる階層utilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_pickle, save_json, load_json
# ゼロ除算を防ぐための定数
EPS = 1e-09

def count_ngram(rows, n, order=True):
  """
  rows中に出現するn-gramの登場回数を数える．

  Parameters
  ------------------
  rows: list of list of str
    各サンプルに対する状態のトレース．
    例えば，
    [
      ['1', '2', '2', '3', '5'],
      ['1', '2', '5'],
      ['1', '2', '4, '7', '3', '4'],
    ]
    みたいな感じ．
  n: int
    n-gramのn．つまり，いくつの状態のまとまりを考えるかを表す数．
  order: bool, default=True
    n-gram内で順番を考慮するかどうか．
    Trueにした場合，例えば(1, 2), (2, 1)は違うbi-gramとして扱うが，Falseだと同じものとして扱う．

  Returns
  ------------------
  ngram_cnt: defaultdict(int)
    n-gram をキーとして，その出現回数を値に持つ辞書．
  """
  ngram_cnt = defaultdict(int)
  for row in rows:
    l = len(row)
    checked = defaultdict(bool)
    for i, s in enumerate(row):
      # i+n > l の場合 n個のかたまりが作れないのでpass
      if (i+n) <= l:
        ngram = row[i:i+n]

        # order=Falseの時はsortしてしまうことで順番の情報をなくす
        if not order:
          ngram.sort()
        
        # そのngramが同じトレース内で既に数えられていた場合は, 重複して数えない
        if not checked[tuple(ngram)]:
            # ngramの出現回数の更新
            ngram_cnt[tuple(ngram)] += 1
            checked[tuple(ngram)] = True
  
  return ngram_cnt

if __name__=="__main__":
  # bootstrap のサンプル数
  B = 10
  # モデルのタイプはLSTMで固定
  model_type = ModelType.LSTM
  
  for ngram_n in range(1, 6):

    # pfa上での各データによる状態遷移の読み込み
    for i in range(B):
      print("========= use bootstrap sample {} for training data =========".format(i))
      for dataset in ['tomita_3', 'tomita_4', 'tomita_7', \
                          DataSet.BP, DataSet.MR, DataSet.IMDB]:
        for k in range(2, 12, 2):
          print("========= dataset={}, k={} =========".format(dataset, k))

          #################################################
          # 成功時，失敗時のトレースをロードし,
          # そこから成功時,失敗時のngramの出現回数を数える
          #################################################

          # pfa spectrumの保存ディレクトリ
          pfa_spec_dir = AbstractData.PFA_SPEC.format(model_type, dataset, i, k, "val")
          # pfaで予測成功時の状態のトレース
          succ_spec_path = os.path.join(pfa_spec_dir, "succ_trace.txt")
          # pfaで予測失敗時の状態のトレース
          fail_spec_path = os.path.join(pfa_spec_dir, "fail_trace.txt")
          # pfaで予測成功/失敗時のトレースを読み込む
          with open(succ_spec_path, 'r') as f:
              succ_rows = f.readlines()
          with open(fail_spec_path, 'r') as f:
              fail_rows = f.readlines()
          # 各行の末尾の改行を取り除き, コンマで配列を切る
          succ_rows = [row.strip().split(',') for row in succ_rows]
          fail_rows = [row.strip().split(',') for row in fail_rows]
          # 予測に成功/失敗時のn-gramを数える
          succ_ngram_cnt = count_ngram(succ_rows, ngram_n)
          fail_ngram_cnt = count_ngram(fail_rows, ngram_n)

          #################################################
          # 成功時,失敗時のngramの出現回数を用いて, 
          # 各ngramに対する疑惑値を計算する
          #################################################
          
          # ngramに対する疑惑値を保存しておく辞書
          ngram_susp_dict = defaultdict(float)
          # 予測に失敗した回数
          Nf = len(fail_rows)
          # 成功時,失敗時のngramの集合
          ngrams = set( list(succ_ngram_cnt.keys()) + list(fail_ngram_cnt.keys()) )
          for ngram in ngrams:
            # そのngramを通って成功,失敗したサンプル数を保存
            Ncs, Ncf = succ_ngram_cnt[ngram], fail_ngram_cnt[ngram] 
            # Ochiaiの式を使って計算する
            ngram_susp_dict[ngram] = Ncf / (np.sqrt(Nf * (Ncf+Ncs)) + EPS)
          # 疑惑値の降順にソートしておく
          ngram_susp_dict = dict(sorted(ngram_susp_dict.items(), key=lambda x:x[1], reverse=True))

          # 計算した疑惑値を所定のディレクトリに保存する
          ngram_susp_dir = PfaSusp.NGRAM_SUSP.format(model_type, dataset, i, k, ngram_n, "val")
          save_pickle(os.path.join(ngram_susp_dir, "ochiai_susp_dict.pkl"), ngram_susp_dict)

          # 疑惑値の上位20件はすぐみれるようにjsonで保存する
          ngram_susp_dict_top20 = {str(k) : ngram_susp_dict[k] for k in list(ngram_susp_dict)[:20]}
          save_json(os.path.join(ngram_susp_dir, "ochiai_susp_top20.json"), ngram_susp_dict_top20)
