import sys
sys.path.append('../') # utilsをインポートするため必要
import argparse
from sklearn.utils import resample
# 異なる階層utilsからのインポート
from utils.constant import *
from utils.help_func import save_pickle, load_pickle

def bootstrap_sampling(data, save_dir, B=10, target="train"):
  """
  bootstrap samplingをB回行う.
  各 bootstrap sample は, {save_dir}/boot_{i}.pkl に保存する.
  
  Parameters
  ------------------
  data: dict
  save_dir: str
  B: int, default=10
    bootstrap samplingを行う回数. 
  target: str, default="train"
    bootstrap samplingを行う対象となるデータ

  """
  target_x, target_y = "{}_x".format(target), "{}_y".format(target)
  for i in range(B):
    boot = {}
    # 無作為復元抽出を行う(標本数はデフォルトなので元のデータ数と同じ)
    boot[target_x], boot[target_y] = resample(data[target_x], data[target_y])
    # save_dirに保存する
    save_pickle(os.path.join(save_dir, "boot_{}.pkl".format(i)), boot) if target=="train" else \
      save_pickle(os.path.join(save_dir, "{}_boot_{}.pkl".format(target, i)), boot)
  print("{} bootstarp samples are saved in dir {}".format(B, save_dir))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", type=str, help='abbrev. of datasets')
  parser.add_argument("target", type=str, help='train or val', default='train', choices=['train', 'val'])
  args = parser.parse_args()
  dataset, target = args.dataset, args.target

  if dataset == 'tomita':
    # tomita7つ
    for tomita_id in range(1, 8, 1):
      # train, val, test分割後データをロード
      split_data = load_pickle(get_path(DataPath.TOMITA.SPLIT_DATA.format(tomita_id)))
      # bootstrapで分割したデータを保存
      bootstrap_sampling(split_data, get_path(DataPath.TOMITA.BOOT_DATA_DIR.format(tomita_id)), B=10, target=target)
  # tomita以外
  else:
    # train, val, test分割後データをロード
    split_data = load_pickle(get_path(getattr(DataPath, dataset.upper()).SPLIT_DATA))
    # bootstrapで分割したデータを保存
    bootstrap_sampling(split_data, get_path(getattr(DataPath, dataset.upper()).BOOT_DATA_DIR), B=10, target=target)