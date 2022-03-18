"""
モデルの学習などに必要なパラメータをデータセット,モデルの種類ごとに定義する
関数名はargs_{dataset}_{model_type}で, これを呼び出すとパラメータのdictが返される
"""

def args_lstm_mr():
  params = {
    "input_size": 300,
    "output_size": 2,
    "hidden_size": 512,
    "num_layers": 1,
    "LEARNING_RATE": 0.01,
    "min_acc": 0.9999,
    "EPOCH": 50, # もともと100だったけど時間かかるので
    "EARLY_STOPPING": False,
  }
  return params

def args_gru_mr():
  return args_lstm_mr()

def args_lstm_bp():
  params = {
    "input_size": 29,
    "output_size": 2,
    "hidden_size": 512,
    "num_layers": 1,
    "LEARNING_RATE": 0.05,
    "min_acc": 0.9999,
    "EPOCH": 100,
    "EARLY_STOPPING": False,
  }
  return params

def args_gru_bp():
  return args_lstm_bp()

def args_lstm_imdb():
  params = {
    "input_size": 300,
    "output_size": 2,
    "hidden_size": 512,
    "num_layers": 1,
    "LEARNING_RATE": 0.01,
    "min_acc": 0.9999,
    "EPOCH": 100,
    # "EPOCH": 1, # for test in loacal
    "EARLY_STOPPING": False,
  }
  return params

def args_gru_imdb():
    return args_lstm_imdb()

def args_lstm_tomita():
  params = {
    "input_size": 3,
    "output_size": 2,
    "hidden_size": 512,
    "num_layers": 1,
    "LEARNING_RATE": 0.01,
    "min_acc": 0.9999,
    "EPOCH": 200,
    # "EPOCH": 1, # for test in loacal
    "EARLY_STOPPING": False,
  }
  return params

def args_gru_tomita():
  return args_lstm_tomita()