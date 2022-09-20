import os
import sys
import glob
import argparse
import torch
import numpy as np
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
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", type=str, help='abbrev. of datasets')
  parser.add_argument("model_type", type=str, help='type of models')
  parser.add_argument("--device", type=str, help='cpu or id of gpu', default='cpu') # gpuならその番号(0,1,...)
  parser.add_argument("--use_clean", type=bool, help='filter stop word or not', default=True)
  parser.add_argument("--start_boot_id", type=int, help='What id of bootstrap sample starts training from.', default=0)
  args = parser.parse_args()

  dataset, model_type = args.dataset, args.model_type
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  isTomita = True if dataset.isdigit() else False
  #  number of bootstrap samplings
  B = 10

  elapsed_times = np.zeros((B))
  for i in range(args.start_boot_id, B):
    print("========= use bootstrap sample {} for training data =========".format(i))
    # モデルファイルが含まれるディレクトリ名を取得
    model_dir = os.path.join(getattr(getattr(TrainedModel, model_type.upper()), dataset.upper()), "boot_{}".format(i)) if not isTomita else \
      os.path.join(getattr(TrainedModel, model_type.upper()).TOMITA.format(dataset), "boot_{}".format(i))
    load_model_path = glob.glob(get_path(os.path.join(model_dir, "*.pkl")))[0]
    # 隠れ状態のトレースをとる
    elapsed_times[i] = make_ori_trace(model_type, dataset, device, load_model_path, i, use_clean=args.use_clean)
    print(f'elapsed for trace extraction: {elapsed_times[i]} [sec.]')
  print(f'mean epalsed for trace extraction in {B} boots: {np.mean(elapsed_times)} [sec.]')