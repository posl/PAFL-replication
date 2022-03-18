import numpy as np
from collections import defaultdict

def get_state_susp(trans_susp_mat):
  """
  各遷移の疑惑値の行列を基にして, 各状態の疑惑値を計算する.
  状態 s から発生する遷移の疑惑値の合計を状態 s の疑惑値として算出する.
  結果は辞書で返し，疑惑値の降順で格納する.
  
  Parameters
  ------------------
  trans_susp_mat: list of list of float
    各遷移の疑惑値の行列.
    get_trans_susp_matの返り値.
  
  Returns
  ------------------
  state_susp_dict: dict
    状態 => その状態の疑惑値 の対応表.
  """
  state_susp_dict = defaultdict(float)
  # 状態 s から発生する遷移の疑惑値の合計を状態 s の疑惑値とする
  for state, state_susp in enumerate(np.sum(trans_susp_mat, axis=1)):
    state_susp_dict[state+1] = state_susp
  # 疑惑値の降順にソートして返す
  state_susp_dict = dict(sorted(state_susp_dict.items(), key=lambda x:x[1], reverse=True))
  return state_susp_dict

