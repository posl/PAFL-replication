import sys
# utilsをインポートするために必要
sys.path.append("../../")
import pprint
import re
import subprocess
import shutil
import numpy as np
import random
import requests
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.time_util import folder_timestamp

def prepare_prism_data(pm_file, num_prop):
  """
  reachability計算用のpmファイルを作成し, カレントディレクトリ直下に保存する.
  元のpmファイルから変更する点は2つ.
  1) pfaの各状態のラベル
  2) pfaの初期状態
  作成されるpmファイル数は, スタート状態と目的状態の組み合わせの数 = (状態数)*(予測ラベル数)個だけある.
  
  Parameters
  ------------------
  pm_file: str
    pfa構築時に出力されるpmファイルのパス
  num_prop: int
    ラベルの数

  Returns
  ------------------
  total_states: int
    pfaの状態数
  tmp_data_path: str
    pmファイル群の保存ディレクトリ
  """
  # ./tmpというディレクトリにreachability計算用のpmファイル群の格納ディレクトリを作成
  TMP_FOLDER = os.path.join(os.getcwd(), "tmp")
  # reachability計算用のpmファイル群の格納ディレクトリ名を定義
  random_id = str(random.randint(1000, 10000))
  folder_id = folder_timestamp()
  tmp_data_path = os.path.join(TMP_FOLDER, folder_id+"_"+random_id)
  # reachability計算用のpmファイル群の格納ディレクトリを作成
  os.makedirs(tmp_data_path)

  # 元のpmファイル(learning_pfaで出力される)をオープン
  with open(pm_file, "r") as fr:
    raw_pm_lines = fr.readlines()
  total_lines = len(raw_pm_lines)
  # pmファイルの"endmodule"の2行後からラベル情報が書かれているので,その行番号をlabel_beginにする
  label_begin = raw_pm_lines.index("endmodule\n") + 2
  # ラベル名の部分だけを正規表現で取り出すためのルール
  label_patter = re.compile(r".*\"(\w+)\".*")
  for idx in range(label_begin, total_lines):
    line = raw_pm_lines[idx]
    # ラベル名の頭に"L"をつけたものを新たなラベル名とする
    new_label = "L" + label_patter.match(line).group(1)
    # 元のラベル名の部分を新たらしいラベル名で置き換える
    raw_pm_lines[idx] = re.sub(r"\"(\w+)\"", '\"' + new_label + '\"', line)

  ptn = re.compile(r".*\[1\.\.(\d+)\].*")
  # pmファイルの4行目は s:[1..5] init 1;　のようになっているのでここから状態数(=5)の部分だけ取り出す
  total_states = int(ptn.match(raw_pm_lines[3]).group(1))
  # スタート状態と目的状態を変えてpmファイルを作成するためのループ
  for start_s in range(1, total_states + 1):
    for prop_id in range(1, num_prop + 1):
      file_name = "s{}_p{}.pm".format(start_s, prop_id)
      # 初期状態(init の後の数字)を変更する
      raw_pm_lines[3] = "s:[1..{}] init {};\n".format(total_states, start_s)
      # ラベル名と初期状態が変更されたpmファイルを保存
      with open(os.path.join(tmp_data_path, file_name), "w") as fw:
        fw.writelines(raw_pm_lines)
  return  total_states, tmp_data_path

def _get_reachability_prob(prism_script, data_path, start_s, prop_id, is_multiclass):
  """
  get reachability prob of a state => a label.
    pfaの状態(start_s)から, 予測ラベル(prop_id)に対応するpfaの状態への, reachability prob(到達確率)を返す.
    pmファイルは, prepare_prism_dataで生成したものを利用し, reachabilityの計算のためにprismスクリプトを実行させる.
  
  Parameters
  ------------------
  prism_script: str
    prismの実行ファイルへのパス
  data_path: str
    必要なpmファイルが格納されているディレクトリのパス
  start_s: int
    開始状態のid
  prop_id: int
    予測ラベルのid
  
  Returns
  ------------------
  reachability_prob: float
    start_sからprop_idに対応する状態への, reachability prob(到達確率)
  """
  # 使用するpmファイルのパス
  pm_file = os.path.join(data_path, "s{}_p{}.pm".format(start_s, prop_id))
  if not is_multiclass:
    property_file = get_path(PROPERTY_FILE_BINARY)
  else:
    property_file = get_path(PROPERTY_FILE_MNIST)
  # send http request to prism server
  headers = {
    'accept': 'application/json',
  }
  params = (
    ('pm_file', os.path.join(*pm_file.split('/')[4:])),
    ('property_file', property_file),
    ('prop_id', prop_id)
  )
  output = requests.get('http://prism_server:8000/prism', headers=headers, params=params).text.strip('"')
  output = output.split(r"\n")
  # prismスクリプトの出力確認用
  # pprint.pprint(output)
  # 出力のうち, reachabilityに相当する部分だけ取り出す
  reachability_prob = float(output[-2].split()[1])
  return reachability_prob

def get_state_reachability(tmp_prism_data_path, num_prop, start_s, is_multiclass):
  """
  get reachability prob of a state => all labels.
    pfaでの開始状態start_sから, 各予測ラベル(に対応するpfaの状態)へのreachability probを計算する
  
  Parameters
  ------------------
  tmp_prism_data_path: str
    必要なpmファイルが格納されているディレクトリのパス
  num_prop: int
    予測ラベルの数
  start_s: int
    開始状態のid
  
  Returns
  ------------------
  row: list of float
    start_sから各予測ラベルへのreachability probを保持する配列.
    rowのサイズ = num_propとなる.
  """
  row = []
  # start_sから各予測ラベルへのreachability probを計算し, rowに追加する
  for prop_id in range(1, num_prop + 1):  # must be 1-index
    ele = _get_reachability_prob(PRISM_SCRIPT, tmp_prism_data_path, start_s, prop_id, is_multiclass)
    row.append(ele)
  return row

def _get_matrix(total_states, num_prop, tmp_prism_data_path):
  """
  get reachability prob of all states => all labels.
    任意の開始状態から任意の予測ラベルへのreachability probを2次元配列の形式で返す.
  
  Parameters
  ------------------
  total_states: int
    pfaの状態数
  num_prop: int
    予測ラベルの数
  tmp_prism_data_path: str
    必要なpmファイルが格納されているディレクトリのパス
  
  Returns
  ------------------
  matrix: list of list of float
    matrix[i][j] = 状態iから予測ラベルj(に対応する状態)へのreachability probが格納された配列
  """
  matrix = []
  # すべての状態を順番に開始状態に設定し, 各予測ラベルへのreachability probを計算し, matrixに追加する
  for start_s in range(1, total_states + 1):
    # 状態start_sから各予測ラベルへのreachability probを1次元配列で取得
    row = get_state_reachability(tmp_prism_data_path, num_prop, start_s)
    # 得られた1次元配列rowをmatrixに追加
    matrix.append(row)
  # numpy配列にしてから返す
  matrix = np.array(matrix)
  return matrix

def reachability_matrix(pm_file, num_prop):
  """
  元のpmファイルと予測ラベルの数から, pfaの各状態から各ラベルへのreachability probを2次元配列の形式で返す.
  以下の3ステップからなる.
  1) prepare_prism_dataを呼び出して, reachability probの計算のためのpmファイル群を生成.
  2) _get_matrixを呼び出してreachability matrixを算出.
  3) matrixを算出後は, prepare_prism_dataで生成されたpmファイル群は不要になるので, 削除する.
  
  Parameters
  ------------------
  pm_file: str
    pfa構築時に出力されるpmファイル(オリジナルのpmファイル)のパス
  num_prop: int
    予測ラベルの数
  
  Returns
  ------------------
  matrix: list of list of float
    matrix[i][j] = 状態iから予測ラベルj(に対応する状態)へのreachability probが格納された配列.
    _get_matrixの返り値.
  """
  # 1) 必要なpmファイル群を生成
  num_states, tmp_data_path = prepare_prism_data(pm_file, num_prop)
  # 2) reachability matrixの計算
  matrix = _get_matrix(num_states, num_prop, tmp_data_path)
  
  # 3) 不要になったディレクトリを, 中のファイルやサブディレクトリごと削除する
  shutil.rmtree(tmp_data_path)
  return matrix
