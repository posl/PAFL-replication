import os
import sys
sys.path.append("../")
import pickle
import json
import numpy as np
import nltk
from nltk.corpus import stopwords

def load_pickle(file_path):
  """
  pklファイルを読み込み, pklファイルに格納されているオブジェクトを返す
  
  Parameters
  ------------------
  file_path: str
    pklファイルのパス

  Returns
  ------------------
  pkl_obj: any
    file_pathのpklファイルに格納されているオブジェクト
  """
  # file_pathが存在しなかったらエラーになる
  with open(file_path, "rb") as f:
    pkl_obj = pickle.load(f)
  return pkl_obj

def save_pickle(file_path, obj, protocol=3):
  """
  オブジェクトobjをpklファイルとしてfile_pathに保存する
  
  Parameters
  ------------------
  file_path: str
    オブジェクトをpklで保存するパス
  obj: any
    保存したいオブジェクト
  protocol: int
    pickleのプロトコルバージョン
  """
  # file_pathが存在しない場合は作成してくれる(load_pickleとの違い)
  parent_path = os.path.split(file_path)[0]
  if not os.path.exists(parent_path):
    os.makedirs(parent_path)
  with open(file_path, "wb") as f:
    pickle.dump(obj, f, protocol=protocol)

def save_json(file_path, obj):
  """
  オブジェクトobjをjsonファイルとしてfile_pathに保存する
  
  Parameters
  ------------------
  file_path: str
    オブジェクトをjsonで保存するパス
  obj: any
    保存したいオブジェクト
  """
  # file_pathが存在しない場合は作成してくれる(load_pickleとの違い)
  parent_path = os.path.split(file_path)[0]
  if not os.path.exists(parent_path):
    os.makedirs(parent_path)
  with open(file_path, "w") as f:
    json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(file_path):
  """
  指定したパスのjsonオブジェクトをdict型として読み込んで返す
  
  Parameters
  ------------------
  file_path: str
    ロードするjsonオブジェクトの保存パス
  
  Returns
  ------------------
  json_obj: dict
    file_pathのjsonファイルを辞書型変数として読み込んだ値
  """
  with open(file_path, "r") as f:
    json_obj = json.load(f)
  return json_obj

def save_readme(parent_path, content, identifier):
  """
  READMEを保存する関数
  
  Parameters
  ------------------
  parent_path: str
    readmeを保存するディレクトリのパス(ルートからのパス)
  content: str
    readmeに記載する内容
  identifier: str
    readmeファイル識別子
    README(identifier)と言う名前になる
  """
  with open(os.path.join(parent_path, "README({})".format(identifier)), "w") as f:
    f.writelines(content)

def make_check_point_folder(task_name, dataset, modelType):
  """
  チェックポイント用のディレクトリを作成する
  ./tmp/{task_name}/{dataset}/{modelType}と言うディレクトリが作られる
  
  Parameters
  ------------------
  task_name: str
  dataset: str
    データセットの種類. bp,mrなど
  modelType: str
    モデルのタイプ. lstm,gruなど
  
  Returns
  ------------------
  check_point_folder: str
    チェックポイント用のディレクトリのパス
  """
  check_point_folder = os.path.join("./tmp", task_name, dataset, modelType)
  if not os.path.exists(check_point_folder):
    os.makedirs(check_point_folder)
  return check_point_folder

def filter_stop_words(sent):
  """
  stop-wordsを文sentから除去する関数
  
  Parameters
  ------------------
  sent: list of str
    除去対象となる文
  
  Returns
  ------------------
  : list of str
    stop-wordsを除去した後の文
  """
  nltk.download('stopwords', quiet=True)
  stop_words = set(stopwords.words('english'))
  special_symbols = set({",", ".", ";", "!", ":", '"', "'", "(", ")", "{", "}", "--"})
  # stop-wordsの集合を生成
  STOP_WORDS = stop_words | special_symbols
  # stop-wordsに含まれない単語のみを残して返す
  return [word for word in sent if word not in STOP_WORDS]

def save_adv_text(file_path, ori_labels, adv_lables, adv_sentences):
  """
  未使用
  """
  paren_path = os.path.split(file_path)[0]
  if not os.path.exists(paren_path):
    os.mkdir(paren_path)
  with open(file_path, "wb") as f:
    pickle.dump({"original_labels": ori_labels,
                  "adv_labels": adv_lables,
                  "adv_sentences": adv_sentences}, f)

def load_adv_text(file_path, shuffle=True):
  """
  未使用
  """
  with open(file_path, "rb") as f:
    data = pickle.load(f)
    ori_labels = data["original_labels"]
    adv_lables = data["adv_labels"]
    adv_sentences = data["adv_sentences"]
  if shuffle:
    idx = [i for i in range(len(ori_labels))]
    np.random.shuffle(idx)
    ori_labels = [ori_labels[i] for i in idx]
    adv_lables = [adv_lables[i] for i in idx]
    adv_sentences = [adv_sentences[i] for i in idx]
  return ori_labels, adv_lables, adv_sentences