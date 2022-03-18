# 統計関係の便利関数たち

import numpy as np

def cohens_d(x1, x2):
  """
  2つのグループのデータx1, x2に対して，Cohenのdを計算する．
  
  Parameters
  ------------------
  x1, x2: list of float (np.ndarrayでもOK)
    2群のデータ

  Returns
  ------------------
  d: float
    x1, x2から計算されたCohenのdの値
  """
  n1 = len(x1)
  n2 = len(x2)
  x1_mean = np.mean(x1)
  x2_mean = np.mean(x2)
  s1 = np.std(x1)
  s2 = np.std(x2)
  s = np.sqrt((n1*np.square(s1) + n2*np.square(s2)) / (n1+n2))
  d = np.abs(x1_mean-x2_mean)/s
  return d

def cliffs_d(x1, x2):
  """
  2つのグループのデータx1, x2に対して，Cliffのdeltaを計算する．
  マンホイットニーの統計量Uと各サンプルサイズを用いて計算することもできるが，その実装はしていない(U検定以外でも対応できるように)．
  
  Parameters
  ------------------
  x1, x2: list of float (np.ndarrayでもOK)
    2群のデータ

  Returns
  ------------------
  d: float
    x1, x2から計算されたCliffのdの値
  """
  n1 = len(x1)
  n2 = len(x2)
  x1, x2 = np.array(x1), np.array(x2)
  d = 0
  for e in x1:
    d += np.sum((e>x2)*1 - (e<x2)*1)
  return d/(n1*n2)

def assess_cohens_d(d):
  """
  Cohenのdの絶対値の大きさを評価する．
  参考: https://cran.r-project.org/web/packages/effsize/effsize.pdf

  Parameters
  ------------------
  d: float
    cohenのdの値

  Returns
  ------------------
  : str
    dの絶対値に応じた評価結果．
    "negligible", "small", "midium", "learge"のいずれか．
  """
  if np.abs(d) < 0.2:
      return "negligible"
  elif np.abs(d) < 0.5:
      return "small"
  elif np.abs(d) < 0.8:
      return "medium"
  else:
      return "large"

def assess_cliffs_d(d):
  """
  Cliffのdeltaの絶対値の大きさを評価する．
  参考: https://cran.r-project.org/web/packages/effsize/effsize.pdf

  Parameters
  ------------------
  d: float
    cliffのdeltaの値

  Returns
  ------------------
  : str
    dの絶対値に応じた評価結果．
    "negligible", "small", "midium", "learge"のいずれか．
  """
  if np.abs(d) < 0.147:
      return "negligible"
  elif np.abs(d) < 0.33:
      return "small"
  elif np.abs(d) < 0.474:
      return "medium"
  else:
      return "large"