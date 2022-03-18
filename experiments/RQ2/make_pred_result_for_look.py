import os
import sys 
# 異なる階層のutils, model_utilsをインポートするために必要
sys.path.append("../../")
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle

if __name__ == "__main__":
  # 定数 metrics名の配列, 列名もこの順番になる
  metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'auc']

  for i in range(10):
    print("\n============== boot_id={} ==============".format(i))

    for n in range(1, 6):

      for theta in [10, 50]:
        # pred_resの保存パスを取得
        pred_res_save_path = ExtractData.PRED_RESULT_FOR_LOOK.format(i, n, theta)
        os.makedirs(Path(pred_res_save_path).parent, exist_ok=True)

        # pred_res_dfを作成
        pred_res_df = pd.DataFrame(index=[], columns=[])
        # dfの列のリスト,  とりあえず最初の2つはdatasetとkにする
        columns = ['dataset', 'k']
        
        for dataset in ['tomita_3', 'tomita_4', 'tomita_7', \
                        DataSet.BP, DataSet.MR, DataSet.IMDB]:
          for k in range(2, 12, 2):
            pred_res_row = {}
            pred_res_row["dataset"] = dataset
            pred_res_row["k"] = str(k)

            # 予測結果のdictを読み出す
            pred_res = defaultdict(dict)
            pred_res['ex'] = load_pickle(ExtractData.EX_PRED.format("lstm", dataset, i, k, n, theta))
            pred_res['rand'] = load_pickle(ExtractData.RAND_PRED.format("lstm", dataset, i, k, n, theta))

            for m in metrics:
              for mode in ["ex", "rand"]:
              # 読み出したdictからpred_res_rowを構成
                key = m + '_' + mode
                # keyがcolumnsに存在しなかったら, 追加する
                if not key in columns:
                  columns.append(key)
                # メトリクスの値(浮動小数点値)を文字列にし,小数点以下3桁のみ取得
                val_s = str(pred_res[mode][m])[:5]
                # 小数点以下の桁数が3桁未満だった場合, 0で埋める
                if len(val_s) < 5:
                  pad_size = 5 - len(val_s)
                  val_s += "0" * pad_size
                pred_res_row[key] = val_s
            
            # 構成したpred_res_rowをdfに追加
            pred_res_df = pred_res_df.append(pred_res_row, ignore_index=True)
        
        # columnsの順番にdfの列順を指定
        pred_res_df = pred_res_df.reindex(columns=columns)
        # thetaごとにpred_res_dfをcsvで保存する
        pred_res_df.to_csv(pred_res_save_path, header=True, index=False, sep=",")