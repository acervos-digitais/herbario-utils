import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from torch import Tensor, nn

class YearData:
  def __init__(self, min_year, max_year, granularity):
    self.min_year = min_year
    self.max_year = max_year
    self.max_class = int((max_year - min_year) / granularity) + 1
    self.granularity = granularity

  def year2class(self, year):
    if year < self.min_year:
      return 0
    elif year >= self.max_year:
      return int((self.max_year - self.min_year) / self.granularity) + 1
    else:
      return int((year - self.min_year) / self.granularity) + 1

  def class2year(self, clsidx):
    if clsidx == 0 or clsidx == self.max_class:
      return 9999
    else:
      return ((clsidx - 1) * self.granularity) + self.min_year


class AverageClassify:
  @classmethod
  def top_k_accuracy(cls, labels, preds, k=1):
    corrects = [1 for l,ps in zip(labels, preds) if l in ps[:k]]
    return len(corrects) / len(labels)

  @classmethod
  def thold_accuracy(cls, labels, preds, thold=10):
    valid_cnt = 0
    correct_cnt = 0
    for idx,pred in enumerate(preds):
      if pred[1] < thold:
        valid_cnt += 1
        if pred[0] == labels[idx]:
          correct_cnt += 1
    
    return 0 if valid_cnt == 0 else (correct_cnt / valid_cnt, valid_cnt / len(labels))

  @classmethod
  def dist_stats(cls, labels, preds):
    correct_idxs = []
    wrong_idxs = []
    for idx,pred in enumerate(preds):
      if pred[0] == labels[idx]:
        correct_idxs.append(idx)
      else:
        wrong_idxs.append(idx)

    correct_dists = preds[correct_idxs][:, 1]
    wrong_dists = preds[wrong_idxs][:, 1]

    print("Correct:", correct_dists.min(), correct_dists.max(), correct_dists.mean())
    if len(wrong_dists) > 0:
      print("Wrong:", wrong_dists.min(), wrong_dists.max(), wrong_dists.mean())

    plt.hist(correct_dists, bins=30)
    plt.title("Correct")
    plt.show()

    plt.hist(wrong_dists, bins=30)
    plt.title("Wrong")
    plt.show()

  def __init__(self, n_averages):
    self.n_averages = n_averages
    self.average_centers = []
    self.idx2class = []

  def fit(self, data):
    classes = np.array([x["class"] for x in data])
    embeddings = np.array([x["embedding"] for x in data])

    self.average_centers = []
    self.idx2class = []

    for mclass in np.sort(np.unique(classes)):
      mclass_embs = embeddings[np.where(classes == mclass)]
      mKMeans = KMeans(n_clusters=min(len(mclass_embs), self.n_averages), random_state=1010)
      mKMeans.fit(mclass_embs)

      for avg_val in mKMeans.cluster_centers_:
        self.average_centers.append(avg_val)
        self.idx2class.append(mclass)

    self.average_centers = np.array(self.average_centers)
    self.idx2class = np.array(self.idx2class)

  def predict(self, data):
    embeddings = np.array([x["embedding"] for x in data])
    dists = euclidean_distances(embeddings, self.average_centers)
    return np.array([[self.idx2class[idx] for idx in ds.reshape(-1).argsort()] for ds in dists])

  def predict_dist(self, data):
    embeddings = np.array([x["embedding"] for x in data])
    dists = euclidean_distances(embeddings, self.average_centers)
    classes_dists = np.array([[[self.idx2class[idx], ds[idx]] for idx in ds.reshape(-1).argsort()] for ds in dists])
    return classes_dists[:, 0]


class ClusterClassify:
  def __init__(self, n_averages, n_clusters=8):
    self.n_averages = n_averages
    self.n_clusters = n_clusters
    self.cluster_average_centers = []
    self.idx2class = []

  def fit(self, data):
    clusters = np.array([x["cluster"] for x in data])

    self.cluster_average_centers = []
    self.idx2class = []

    for mcluster in np.sort(np.unique(clusters)):
      cluster_data = np.array(data)[np.where(clusters == mcluster)]
      cluster_classes = np.array([x["class"] for x in cluster_data])
      cluster_embeddings = np.array([x["embedding"] for x in cluster_data])

      for mclass in np.sort(np.unique(cluster_classes)):
        mclass_embs = cluster_embeddings[np.where(cluster_classes == mclass)]
        mKMeans = KMeans(n_clusters=min(len(mclass_embs), self.n_averages), random_state=1010)
        mKMeans.fit(mclass_embs)

        for avg_val in mKMeans.cluster_centers_:
          self.cluster_average_centers.append(avg_val)
          self.idx2class.append(mclass)

    self.cluster_average_centers = np.array(self.cluster_average_centers)
    self.idx2class = np.array(self.idx2class)

  def predict(self, data):
    embeddings = np.array([x["embedding"] for x in data])
    dists = euclidean_distances(embeddings, self.cluster_average_centers)
    return np.array([[self.idx2class[idx] for idx in ds.reshape(-1).argsort()] for ds in dists])


class SKClassify:
  @classmethod
  def thold_accuracy(cls, labels, preds, thold=0.8):
    valid_cnt = 0
    correct_cnt = 0
    for idx,pred in enumerate(preds):
      if pred[1] > thold:
        valid_cnt += 1
        if pred[0] == labels[idx]:
          correct_cnt += 1
    
    return 0 if valid_cnt == 0 else (correct_cnt / valid_cnt, valid_cnt / len(labels))

  def fit(self, data):
    classes = np.array([x["class"] for x in data])
    embeddings = np.array([x["embedding"] for x in data])

    if self.pca:
      self.mCC.fit(self.pca.fit_transform(embeddings), classes)
    else:
      self.mCC.fit(embeddings, classes)

  def predict(self, data):
    embeddings = np.array([x["embedding"] for x in data])

    if self.pca:
      return self.mCC.predict(self.pca.transform(embeddings))
    else:
      return self.mCC.predict(embeddings)

  def predict_prob(self, data):
    embeddings = np.array([x["embedding"] for x in data])

    if self.pca:
      embeddings = self.pca.transform(embeddings)

    probs = self.mCC.predict_proba(embeddings)
    classes_probs = np.array([[[idx, ps[idx]] for idx in (-ps).reshape(-1).argsort()] for ps in probs])
    return classes_probs[:, 0]

class RFClassify(SKClassify):
  def __init__(self, n_components=None):
    self.mCC = RandomForestClassifier()
    self.pca = PCA(n_components=n_components) if n_components else None

class KNNClassify(SKClassify):
  def __init__(self, n_neighbors=5, n_components=None):
    self.mCC = KNeighborsClassifier(n_neighbors=n_neighbors)
    self.pca = PCA(n_components=n_components) if n_components else None

class SVClassify(SKClassify):
  def __init__(self, C=1.0, n_components=None):
    self.mCC = SVC(C=C, probability=True)
    self.pca = PCA(n_components=n_components) if n_components else None

class MLPClassify(SKClassify):
  def __init__(self, n_components=None):
    self.mCC = MLPClassifier(hidden_layer_sizes=(128))
    self.pca = PCA(n_components=n_components) if n_components else None

class GaussianProcessClassify(SKClassify):
  def __init__(self, n_components=None):
    self.mCC = GaussianProcessClassifier()
    self.pca = PCA(n_components=n_components) if n_components else None

class GaussianBayesClassify(SKClassify):
  def __init__(self, n_components=None):
    self.mCC = GaussianNB()
    self.pca = PCA(n_components=n_components) if n_components else None

class SGDClassify(SKClassify):
  def __init__(self, n_components=None):
    self.mCC = SGDClassifier(loss="modified_huber")
    self.pca = PCA(n_components=n_components) if n_components else None


class TorchClassify:
  def __init__(self, lr=1e-6, epochs=32):
    self.learning_rate = lr
    self.epochs = epochs
    self.loss_fn = nn.CrossEntropyLoss()

  def fit(self, data):
    classes = Tensor([x["class"] for x in data]).long()
    embeddings = Tensor([x["embedding"] for x in data])

    self.model =  nn.Sequential(
      nn.Dropout(0.35),
      nn.Linear(embeddings.shape[1], embeddings.shape[1] // 2),
      nn.BatchNorm1d(embeddings.shape[1] // 2),
      nn.LayerNorm(embeddings.shape[1] // 2),
      nn.ReLU(),

      nn.Dropout(0.35),
      nn.Linear(embeddings.shape[1] // 2, len(classes.unique())),
    )

    optim = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    for e in range(self.epochs):
      optim.zero_grad()
      classes_pred = self.model(embeddings)
      loss = self.loss_fn(classes_pred, classes)
      loss.backward()
      optim.step()
      if e % (self.epochs // 8) == 0:
        print(f"Epoch: {e} loss: {loss.item():.4f}")

  def predict(self, data):
    embeddings = Tensor([x["embedding"] for x in data])
    self.model.eval()
    with torch.no_grad():
      class_pred = self.model(embeddings).argmax(dim=1)
    return [l.item() for l in class_pred]

  def predict_prob(self, data):
    embeddings = Tensor([x["embedding"] for x in data])
    self.model.eval()
    with torch.no_grad():
      probs = self.model(embeddings)
      classes_probs = np.array([[[idx, ps[idx]] for idx in (-ps).reshape(-1).argsort()] for ps in probs])
    return classes_probs[:, 0]
