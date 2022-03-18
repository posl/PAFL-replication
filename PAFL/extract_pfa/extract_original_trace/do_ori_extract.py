import os
import sys
import glob
import torch
sys.path.append("../../") # extract_pfa/extract_original_traceをインポートするために必要
sys.path.append("../../../") # model_utilsをインポートするために必要
# 同階層のextract_original_traceからインポート
from extract_pfa.extract_original_trace.extract_ori_trace import make_ori_trace
# 異なる階層のmodel_utilsからインポート
from model_utils.model_util import get_model_file
# 異なる階層のutilsからインポート
from utils.constant import *

if __name__=="__main__":
  # コマンドライン引数でデータセットやモデルタイプ,GPUの情報を入力
  dataset = sys.argv[1]
  isTomita = True if dataset.isdigit() else False
  model_type = sys.argv[2]
  use_clean = int(sys.argv[3])
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  # device = 'cpu'
  #  number of bootstrap samplings
  B = 10

  for i in range(B):
    print("========= use bootstrap sample {} for training data =========".format(i))
    # モデルファイルが含まれるディレクトリ名を取得
    model_dir = os.path.join(getattr(getattr(TrainedModel, model_type.upper()), dataset.upper()), "boot_{}".format(i)) if not isTomita else \
      os.path.join(getattr(TrainedModel, model_type.upper()).TOMITA.format(dataset), "boot_{}".format(i))
    load_model_path = glob.glob(get_path(os.path.join(model_dir, "*.pkl")))[0]
    # 隠れ状態のトレースをとる
    make_ori_trace(model_type, dataset, device, load_model_path, i, use_clean=use_clean)