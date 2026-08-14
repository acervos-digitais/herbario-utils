import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from random import sample, seed

from scipy.optimize import minimize
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score as skl_silhouette_score, silhouette_samples as skl_silhouette_samples
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

from umap import UMAP

from PIL import Image as PImage

def reorder_by_cluster_size(X, labels, centers):
  order = np.argsort(np.bincount(labels))
  centers_ordered = centers[order]
  mapping = np.empty_like(order)
  mapping[order] = np.arange(len(order))
  labels_ordered = mapping[labels]
  return X, labels_ordered, centers_ordered


def raw_kmeans(emb_raw, n_clusters=8, normalize=False):
  mCluster = KMeans(n_clusters=n_clusters, random_state=1010)

  emb_pre = Normalizer().fit_transform(emb_raw) if normalize else emb_raw
  labels = mCluster.fit_predict(emb_pre)

  return reorder_by_cluster_size(emb_pre, labels, mCluster.cluster_centers_)


def pca_kmeans(emb_raw, n_clusters=8, n_components=128, normalize=False):
  n_components = min(n_components, len(emb_raw))
  mPCA = PCA(n_components=n_components, random_state=10)
  mCluster = KMeans(n_clusters=n_clusters, random_state=1010)

  emb_pre = Normalizer().fit_transform(emb_raw) if normalize else emb_raw
  emb_reduced = mPCA.fit_transform(emb_pre)
  labels = mCluster.fit_predict(emb_reduced)

  return reorder_by_cluster_size(emb_reduced, labels, mCluster.cluster_centers_)


def tsne_kmeans(emb_raw, n_clusters=8, n_components=3, perplexity=30, normalize=False):
  mTSNE = TSNE(n_components=n_components, perplexity=perplexity, random_state=10)
  mCluster = KMeans(n_clusters=n_clusters, random_state=1010)

  emb_pre = Normalizer().fit_transform(emb_raw) if normalize else emb_raw
  emb_reduced = mTSNE.fit_transform(emb_pre)
  labels = mCluster.fit_predict(emb_reduced)

  return reorder_by_cluster_size(emb_reduced, labels, mCluster.cluster_centers_)


def umap_kmeans(emb_raw, n_clusters=8, n_components=64, n_neighbors=100, normalize=False):
  mUMAP = UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=10, transform_seed=1010, metric="cosine")
  mCluster = KMeans(n_clusters=n_clusters, random_state=1010)

  emb_pre = Normalizer().fit_transform(emb_raw) if normalize else emb_raw
  emb_reduced = mUMAP.fit_transform(emb_pre)
  labels = mCluster.fit_predict(emb_reduced)

  return reorder_by_cluster_size(emb_reduced, labels, mCluster.cluster_centers_)


def pca_agglo(emb_raw, n_clusters=8, n_components=128):
  n_components = min(n_components, len(emb_raw))
  mPCA = PCA(n_components=n_components, random_state=10)
  mCluster = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")

  emb_reduced = mPCA.fit_transform(emb_raw)
  labels = mCluster.fit_predict(emb_reduced)
  centers = np.array([emb_reduced[labels == i].mean(axis=0) for i in range(n_clusters)])

  return reorder_by_cluster_size(emb_reduced, labels, centers)


def cluster_center_from_dists(known_points, known_dists):
  def error_function(target_pos, points, distances):
    calc_dists = np.linalg.norm(points - target_pos, axis=1)
    return np.sum((calc_dists - distances) ** 2)

  initial_guess = np.mean(known_points, axis=0)

  result = minimize(
    fun=error_function,
    x0=initial_guess,
    args=(known_points, known_dists),
    method="Nelder-Mead"
  )

  if result.success:
    return result.x
  else:
    raise Exception(result.message)


def cluster_centers_from_clusters_info(clusters_info, embedding_data):
  nclusters = len(clusters_info["clusters"]["descriptions"]["gemma3"]["en"])
  centers = []
  for cid in range(0, nclusters):
    cluster_images = { iid:iinfo for iid,iinfo in clusters_info["images"].items() if iinfo["cluster"] == cid }
    seed(101010)
    img_ids = sample(cluster_images.keys(), k=64)
    img_locs = [embedding_data[iid] for iid in img_ids]
    img_dists = np.array([clusters_info["images"][iid]["distances"] for iid in img_ids])[:, cid]
    center = cluster_center_from_dists(img_locs, img_dists)
    centers.append([round(x,6) for x in center.tolist()])
  return centers


def get_cluster_variances(X, y):
  num_clusters = len(np.unique(y))
  variance = []
  for i in range(num_clusters):
    ith_cluster_embeddigs = X[y == i]
    mPCA = PCA(n_components=32, random_state=10)
    _ = mPCA.fit_transform(ith_cluster_embeddigs)
    cumulative_variance = mPCA.explained_variance_ratio_.cumsum()
    variance.append([round(v,6) for v in cumulative_variance.tolist()[:8]])
  return variance


def plot_clusters(clusters, pcas, title="", color_clusters=True):
  sizes = [0 if c < 0 else 24 for c in clusters]
  dims = pcas.shape[1]
  plot_dims = min(dims, 3)

  plot_params = {
    "marker": "o",
    "s": sizes,
    "alpha": 0.35,
    "edgecolors": "none"
  }

  if color_clusters:
    plot_params["c"] = clusters
    plot_params["cmap"] = "tab10"

  for i in range(plot_dims):
    for j in range(i+1, plot_dims):
      plt.scatter(pcas[:,i], pcas[:,j], **plot_params)
      plt.title(title)
      plt.show()

  # 3D
  if dims > 2:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pcas[:,0], pcas[:,1], pcas[:,2], **plot_params)
    ax.set_title(title)
    plt.show()

def get_silhouette_scores(X, y):
  mplot, scores = create_silhouette_plot(X, y)
  mplot.close()
  return scores

def save_silhouette_plot(X, y, file_path, title):
  mplot, _ = create_silhouette_plot(X, y, title=title)
  mplot.savefig(file_path, dpi=300, bbox_inches="tight")
  mplot.close()

def display_silhouette_plot(X, y, title):
  mplot, _ = create_silhouette_plot(X, y, title=title)
  mplot.show()

def create_silhouette_plot(X, y, title=None):
  sample_silhouette_values = skl_silhouette_samples(X, y)
  silhouette_average = skl_silhouette_score(X, y)
  num_clusters = len(np.unique(y))
  maxx = round(sample_silhouette_values.max() / 0.2) * 0.2
  plot_title = "Silhouette Plot" if title is None else title

  y_lower = 10
  raw_scores = [round(s,4) for s in sample_silhouette_values.tolist()]
  processed_scores = []
  for i in range(num_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[y == i]
    ith_cluster_silhouette_values.sort()
    processed_scores.append({
      "min": float(ith_cluster_silhouette_values.min()),
      "max": float(ith_cluster_silhouette_values.max()),
      "mean": float(ith_cluster_silhouette_values.mean()),
      "std": float(ith_cluster_silhouette_values.std()),
      "median": float(np.median(ith_cluster_silhouette_values)),
    })

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / num_clusters)
    plt.fill_betweenx(
      np.arange(y_lower, y_upper),
      0,
      ith_cluster_silhouette_values,
      facecolor=color,
      edgecolor=color,
      alpha=0.7,
    )

    # Label the silhouette plots with their cluster numbers at the middle
    plt.text(-maxx / 10, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10

  plt.title(plot_title)
  plt.xlabel("Silhouette coefficient values")
  plt.ylabel("Cluster label")

  # The vertical line for average silhouette score of all the values
  plt.axvline(x=silhouette_average, color="red", linestyle="--")

  plt.yticks([])
  plt.xlim([min(-0.1, sample_silhouette_values.min()), sample_silhouette_values.max()])
  plt.xticks([-0.1] + list(np.arange(0, maxx+0.1, 0.2)))
  return plt, {"raw": raw_scores, "processed": processed_scores}


def visualize_pca_clusters(raw_embeddings, image_paths, n_clusters=8, grid_dim=8):
  m_emb, m_clusters, m_centers = pca_kmeans(raw_embeddings, n_clusters=n_clusters)
  visualize_clusters(m_emb, m_clusters, m_centers, image_paths, grid_dim=grid_dim)

def visualize_tsne_clusters(raw_embeddings, image_paths, n_clusters=8, grid_dim=8):
  m_emb, m_clusters, m_centers = tsne_kmeans(raw_embeddings, n_clusters=n_clusters)
  visualize_clusters(m_emb, m_clusters, m_centers, image_paths, grid_dim=grid_dim)

def visualize_umap_clusters(raw_embeddings, image_paths, n_clusters=8, grid_dim=8):
  m_emb, m_clusters, m_centers = umap_kmeans(raw_embeddings, n_clusters=n_clusters)
  visualize_clusters(m_emb, m_clusters, m_centers, image_paths, grid_dim=grid_dim)

def visualize_clusters(m_emb, m_clusters, m_centers, image_paths, grid_dim=8):
  for c in np.unique(m_clusters):
    cluster_center = m_centers[c]
    cluster_idxs = np.where(m_clusters == c)[0]
    cluster_pcas = m_emb[cluster_idxs]
    pca_center_dists = np.linalg.norm(cluster_pcas - cluster_center, axis=1)
    cluster_idxs_sorted = cluster_idxs[pca_center_dists.argsort()]

    fig, axes = plt.subplots(nrows=grid_dim, ncols=grid_dim)
    fig.set_size_inches(10, 10)
    fig.set_dpi(72)

    fig.suptitle(f"Cluster {c}")
    for ciidx, ax in enumerate(axes.flat):
      ax.axis("off")
      if ciidx < len(cluster_idxs_sorted):
        iidx = cluster_idxs_sorted[ciidx]
        img = PImage.open(image_paths[iidx]).convert("RGB")
        img = img.resize((128,128))
        ax.imshow(img)

    plt.tight_layout()
    plt.show()
