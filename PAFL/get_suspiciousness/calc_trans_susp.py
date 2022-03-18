import numpy as np

# ゼロ除算を防ぐための定数
EPS = 1e-09

def count_succ_fail(pred_info):
  """
  get_pfa_spectrumで出力されたpred_infoからPFAの予測の分割表の部分を取り出す.
  取り出した分割表から, 予測に成功したサンプルの数と失敗したサンプルの数を計算し, 返す.
  
  Parameters
  ------------------
  pred_info: dict
    get_pfa_spectrumで出力されたpred_info
  
  Returns
  ------------------
  Ns: int
    # 予測に成功したサンプル数
  Nf: int
    予測に失敗したサンプル数
  """
  pfa_conti = pred_info['pfa_conti_table']
  Ns = pfa_conti[0][0] + pfa_conti[1][1]
  Nf = pfa_conti[0][1] + pfa_conti[1][0]
  return Ns, Nf

def calc_ochiai(Ns, Nf, Ncs, Ncf, Nus, Nuf):
  """
  SBFLのOchiaiの計算式に基づいて遷移ごとの疑惑値の行列を返す.
  
  Parameters
  ------------------
  Ns: int
    pfaでの予測に成功したテストデータのサンプル数
  Nf: int
    pfaでの予測に失敗したテストデータのサンプル数
  Ncs: list of list of int
    Ncs[i][i] = 状態iから状態jへの遷移を通り, 予測に成功したテストデータのサンプル数 となる行列.
  Ncf: list of list of int
    Ncf[i][i] = 状態iから状態jへの遷移を通り, 予測に失敗したテストデータのサンプル数 となる行列.
  Nus: list of list of int
    Nus[i][i] = 状態iから状態jへの遷移を通らず, 予測に成功したテストデータのサンプル数 となる行列.
  Nuf: list of list of int
    Nuf[i][i] = 状態iから状態jへの遷移を通らず, 予測に失敗したテストデータのサンプル数 となる行列.
  
  Returns
  ------------------
  Ochiai: list of list of float
    Ochiai[i][j] = 状態iから状態jへの遷移に対する疑惑値 となるような行列. 
  """
  Ochiai = Ncf / (np.sqrt(Nf * (Ncf+Ncs)) + EPS)
  return Ochiai

def calc_tarantula(Ns, Nf, Ncs, Ncf, Nus, Nuf):
  """
  SBFLのTarantilaの計算式に基づいて遷移ごとの疑惑値の行列を返す.
  
  Parameters
  ------------------
  Ns: int
    pfaでの予測に成功したテストデータのサンプル数
  Nf: int
    pfaでの予測に失敗したテストデータのサンプル数
  Ncs: list of list of int
    Ncs[i][i] = 状態iから状態jへの遷移を通り, 予測に成功したテストデータのサンプル数 となる行列.
  Ncf: list of list of int
    Ncf[i][i] = 状態iから状態jへの遷移を通り, 予測に失敗したテストデータのサンプル数 となる行列.
  Nus: list of list of int
    Nus[i][i] = 状態iから状態jへの遷移を通らず, 予測に成功したテストデータのサンプル数 となる行列.
  Nuf: list of list of int
    Nuf[i][i] = 状態iから状態jへの遷移を通らず, 予測に失敗したテストデータのサンプル数 となる行列.
  
  Returns
  ------------------
  Tarantula: list of list of float
    Tarantula[i][j] = 状態iから状態jへの遷移に対する疑惑値 となるような行列. 
  """
  Tarantula = (Ncf/Nf) / ((Ncf/Nf) + (Ncs/Ns) + EPS)
  return Tarantula

def calc_dstar(Ns, Nf, Ncs, Ncf, Nus, Nuf, star=3):
  """
  SBFLのDStarの計算式に基づいて遷移ごとの疑惑値の行列を返す.
  
  Parameters
  ------------------
  Ns: int
    pfaでの予測に成功したテストデータのサンプル数
  Nf: int
    pfaでの予測に失敗したテストデータのサンプル数
  Ncs: list of list of int
    Ncs[i][i] = 状態iから状態jへの遷移を通り, 予測に成功したテストデータのサンプル数 となる行列.
  Ncf: list of list of int
    Ncf[i][i] = 状態iから状態jへの遷移を通り, 予測に失敗したテストデータのサンプル数 となる行列.
  Nus: list of list of int
    Nus[i][i] = 状態iから状態jへの遷移を通らず, 予測に成功したテストデータのサンプル数 となる行列.
  Nuf: list of list of int
    Nuf[i][i] = 状態iから状態jへの遷移を通らず, 予測に失敗したテストデータのサンプル数 となる行列.
  star: float
    DStarの計算の変数. 論文でstar=3が使われているのでそれを流用.
  
  Returns
  ------------------
  DStar: list of list of float
    DStar[i][j] = 状態iから状態jへの遷移に対する疑惑値 となるような行列. 
  """
  DStar = (Ncf**star) / (Ncs + Nuf + EPS)
  return DStar

def get_trans_susp_mat(Ns, Nf, succ_mat, fail_mat, sbfl_method):
  """
  各遷移の疑惑値を計算し, 行列形式で返す
  疑惑値の計算のための関数は引数で受け取る(この関数は高階関数).
  
  Parameters
  ------------------
  Ns: int
    pfaでの予測に成功したテストデータのサンプル数
  Nf: int
    pfaでの予測に失敗したテストデータのサンプル数
  succ_mat: list of list of int
    succ_mat[i][i] = 状態iから状態jへの遷移を通り, 予測に成功したテストデータのサンプル数 となる行列.
  fail_mat: list of list of int
    fail_mat[i][i] = 状態iから状態jへの遷移を通り, 予測に失敗したテストデータのサンプル数 となる行列.
  sbfl_method: function(int, int, list of list of int, list of list of int) -> list of list of float
    上記4つの変数から各状態遷移の疑惑値を計算する関数. Ochiai, DStarなど. 

  Returns
  ------------------
  trans_susp_mat: list of list of float
    trans_susp_mat[i][j] = 状態iから状態jへの遷移に対する疑惑値 となるような行列. 
    サイズは (pfaの状態数) * (pfaの状態数) の行列になる. 
  """
  trans_susp_mat = sbfl_method(Ns, Nf, Ncs=succ_mat, Ncf=fail_mat, Nus=Ns-succ_mat, Nuf=Nf-fail_mat)
  return trans_susp_mat

