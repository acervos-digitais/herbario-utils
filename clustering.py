import numpy as np

from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from PIL import Image as PImage

def raw_kmeans(emb_raw, n_clusters=8):
  mCluster = KMeans(n_clusters=n_clusters, random_state=1010)

  emb_clusters = mCluster.fit_predict(StandardScaler().fit_transform(emb_raw))

  return emb_raw, emb_clusters, mCluster.cluster_centers_


def pca_kmeans(emb_raw, n_clusters=8, n_components=128):
  n_components = min(n_components, len(emb_raw))
  mPCA = PCA(n_components=n_components, random_state=10)
  mCluster = KMeans(n_clusters=n_clusters, random_state=1010)

  emb_reduced = mPCA.fit_transform(StandardScaler().fit_transform(emb_raw))
  emb_clusters = mCluster.fit_predict(emb_reduced)

  return emb_reduced, emb_clusters, mCluster.cluster_centers_


def tsne_kmeans(emb_raw, n_clusters=8, n_components=3, perplexity=30):
  mTSNE = TSNE(n_components=n_components, perplexity=perplexity, random_state=10)
  mCluster = KMeans(n_clusters=n_clusters, random_state=1010)

  emb_reduced = mTSNE.fit_transform(StandardScaler().fit_transform(emb_raw))
  emb_clusters = mCluster.fit_predict(emb_reduced)

  return emb_reduced, emb_clusters, mCluster.cluster_centers_


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


def visualize_pca_clusters(raw_embeddings, image_paths, n_clusters=8, grid_dim=8):
  m_emb, m_clusters, m_centers = pca_kmeans(raw_embeddings, n_clusters=n_clusters)
  visualize_clusters(m_emb, m_clusters, m_centers, image_paths, grid_dim=grid_dim)

def visualize_tsne_clusters(raw_embeddings, image_paths, n_clusters=8, grid_dim=8):
  m_emb, m_clusters, m_centers = tsne_kmeans(raw_embeddings, n_clusters=n_clusters)
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
