"""
クラスタリングのクラスを定義する.
クラスタリングの抽象クラスと, KMeansクラスタ, 凝集型階層クラスタリングを定義している.
"""

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier

class Partitioner(object):
  """
  partitioner(クラスタリング関数)の抽象クラス.
  3つのインタフェース(メソッド)を持つ.
  1) fit
    クラスタリングの学習を行う
  2) predict
    データを新たに入力し, 学習済みpartitionerを用いてクラスタに割り当てる
  3) get_fit_labels
    クラスタリングの(学習に用いた)各入力点に対する分類ラベルを取り出す
  """
  def fit(self, X):
    pass

  def predict(self, X):
    pass

  def get_fit_labels(self):
    pass


class Kmeans(Partitioner):
  """
  KMeansクラスタリングのクラス.
  sklearn.cluster.KMeansをベースにしている.
  """
  def __init__(self, n_clusters):
    self.kmeans = KMeans(n_clusters=n_clusters)

  def fit(self, X):
    self.kmeans = self.kmeans.fit(X)

  def predict(self, X):
    return self.kmeans.predict(X)

  def get_fit_labels(self):
    return list(self.kmeans.labels_)

class EHCluster(Partitioner):
  """
  凝集型階層クラスタリングのクラス.
  実際はKMeansしか使っていないが.
  """
  def __init__(self, n_clusters, n_neighbors=5):
    self.clustering = AgglomerativeClustering(n_clusters=n_clusters)
    self.neigh = KNeighborsClassifier(n_neighbors=n_neighbors)

  def fit(self, X):
    self.clustering.fit(X)
    self.neigh.fit(X, self.clustering.labels_)

  def predict(self, X):
    return self.neigh.predict(X)

  def get_fit_labels(self):
    return list(self.clustering.labels_)