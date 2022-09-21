import os
import sys, glob
import argparse
# 異なる階層のutilsをインポートするために必要
sys.path.append("../../")
import numpy as np
import pandas as pd
import math
import re
import random
from collections import defaultdict
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_pickle, load_json, save_json
from utils.susp_score import *
# 異なる階層のmodel_utilsからインポート
from model_utils.model_util import load_model, sent2tensor
from model_utils import train_args

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
sns.set_context('paper', 1.5)
import warnings
warnings.simplefilter('ignore')

if __name__ == '__main__':
  for model_type in ['srnn', 'gru', 'lstm']:
    # ties.pyを実行して得られたarrayをロードする
    res = load_pickle(f'./tie_res_{model_type}.pkl')
    df_k = pd.DataFrame(columns=['dataset', '2', '4', '6', '8', '10'])
    df_k['dataset'] = ['Tomita3', 'Tomita4', 'Tomita7', 'BP', 'RTMR', 'IMDB', 'TOXIC', 'MNIST']
    df_k.iloc[:, [1,2,3,4,5]] = res.mean(axis=(1, 2, 3))
    display(df_k)
    df_n = pd.DataFrame(columns=['dataset', '1', '2', '3', '4', '5'])
    df_n['dataset'] = ['Tomita3', 'Tomita4', 'Tomita7', 'BP', 'RTMR', 'IMDB', 'TOXIC', 'MNIST']
    df_n.iloc[:, [1,2,3,4,5]] = res.mean(axis=(1, 2, 4))
    display(df_n)
    np.savetxt(f'./tie_rate_per_k_{model_type}.csv', df_k.iloc[:, 1:], delimiter=',', fmt='%.5f')
    np.savetxt(f'./tie_rate_per_n_{model_type}.csv', df_n.iloc[:, 1:], delimiter=',', fmt='%.5f')