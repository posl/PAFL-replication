import os
import sys
sys.path.append("../") # utilsをインポートするため必要
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
# 同階層data_preparationからのインポート
from data_preparation.preparation_helper import *
# 異なる階層utilsからのインポート
from utils.help_func import save_pickle, load_pickle, filter_stop_words
from utils.constant import DataPath, get_path

def divide_mnist():
  data = {}
  # 処理済みデータの保存先
  save_path = get_path(DataPath.MNIST.PROCESSED_DATA)
  # train, testデータをロード
  train_set = datasets.MNIST(
    root = get_path(DataPath.MNIST.RAW_DATA),
    train = True,
    download = True,
    transform=transforms.ToTensor()
  )
  test_set = datasets.MNIST(
    root = get_path(DataPath.MNIST.RAW_DATA),
    train = False,
    download = True,
    transform=transforms.ToTensor()
  )
  data["train_x"], data["train_y"] = train_set.data, train_set.targets
  data["test_x"], data["test_y"] = test_set.data, test_set.targets
  data["classes"] = list(set(np.array(train_set.targets)))

  save_pickle(save_path, data)
  print("saved in {}".format(save_path))
  return data