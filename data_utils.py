import json

import numpy as np

from os import path
from sklearn.metrics.pairwise import euclidean_distances

from clustering import tsne_kmeans
from models.LlamaVision import LlamaVision
from models.SigLip2 import SigLip2

STOP_WORDS = {
  "en": ["form", "forms", "subject", "subjects", "figure", "figures", "observer", "observers", "viewer", "viewers", "background", "apparition"],
  "pt": ["formulários", "formulário", "hóspede", "hóspedes", "estudioso", "estudiosos", "observador", "observadores", "figura", "figuras", "colono", "colonos", "caracteres", "subcrescimento", "sendo", "marfim", "assinatura", "pescoço", "ver", "vénus"]
}

REPLACE = {
  "en": {},
  "pt": {
    "acidente vascular cerebral": "pincelada",
    "ainda vida": "natureza morta",
  }
}

def get_caption_words(data_path, model="gemma3", lang="en", categories=["all"], return_counts=False):
  with open(data_path, "r") as ifp:
    obj_data = json.load(ifp)

  all_words = {}
  for obj in obj_data.values():
    for obj_cat in categories:
      for raw_word in obj["captions"][model][lang].get(obj_cat, []):
        word = raw_word.lower().replace(".", "")
        word = REPLACE[lang].get(word, word)
        if word not in STOP_WORDS[lang]:
          all_words[word] = all_words.get(word, 0) + 1

  word_cnts = [[w, c] for w,c in all_words.items()]
  words = sorted(word_cnts, key=lambda x:x[1], reverse=True)

  if return_counts:
    return words
  else:
    return [w for w,_ in words]


def export_preload_data(data_prefix, fields, out_file_name="preload.json"):
  input_file_path = f"./metadata/json/{data_prefix}_processed.json"
  output_file_path = f"./metadata/json/{data_prefix}_{out_file_name}"

  with open(input_file_path, "r") as ifp:
    obj_data = json.load(ifp)

  preload_data = { f: {} for f in fields }

  for k,v in obj_data.items():
    for f in fields:
      f_vals = v[f]
      if type(f_vals) != list:
        f_vals = [f_vals]

      for val in f_vals:
        if type(val) == dict and "label" in val:
          val = val["label"]
        if val not in preload_data[f]:
          preload_data[f][val] = []
        preload_data[f][val].append(k)

  for k in list(preload_data.keys()):
    if k[-1] != "s":
      preload_data[k+"s"] = preload_data.pop(k)

  with open(output_file_path, "w") as ofp:
    json.dump(preload_data, ofp, separators=(",",":"), sort_keys=True, ensure_ascii=False)


class Clusterer:
  def __init__(self, data_prefix, images_dir_path):
    self.data_prefix = data_prefix
    self.images_dir_path = images_dir_path
    self.data_file_path = f"./metadata/json/{data_prefix}_processed.json"
    self.embedding_file_path = f"./metadata/json/{data_prefix}_embeddings.json"
    self.llama = None
    self.siglip = None

    with open(self.embedding_file_path, "r") as ifp:
      self.embedding_data = json.load(ifp)

    self.cluster_data = {}

  def export_clusters(self, out_file_name, embedding_model="siglip2", min_nc=2, max_nc=17, step_nc=2, describe="vlm", **describe_params):
    ids = np.array(list(self.embedding_data.keys()))
    embeddings = np.array([v[embedding_model] for v in self.embedding_data.values()])

    for nc in range(min_nc, max_nc, step_nc):
      print(nc, "clusters...")
      embs, clusters, centers = tsne_kmeans(embeddings, n_clusters=nc)
      cluster_distances = euclidean_distances(centers, embs)
      id_idxs_by_distance = cluster_distances.argsort(axis=1)
      ids_by_distance = ids[id_idxs_by_distance]

      i_c_d = zip(ids.tolist(), clusters.tolist(), cluster_distances.T.tolist())

      if describe == "vlm":
        descriptions = self.describe_by_vlm(ids_by_distance, **describe_params)
      else:
        descriptions = self.describe_by_siglip2(ids_by_distance, **describe_params)

      self.cluster_data[nc] = {
        "images": {id: {"cluster": c, "distances": [round(d,6) for d in ds]} for  id,c,ds in i_c_d},
        "clusters": {"descriptions": descriptions}
      }

    out_file_path = f"./metadata/json/{self.data_prefix}_{out_file_name}"
    with open(out_file_path, "w") as ofp:
      json.dump(self.cluster_data, ofp, separators=(",",":"), sort_keys=True, ensure_ascii=False)


  def describe_by_vlm(self, ids_by_distance, top_images=10, num_images=10):
    if self.llama == None:
      self.llama = LlamaVision()

    idx_end = top_images
    idx_step = int(top_images // num_images)
    ids_to_describe = ids_by_distance[:, :idx_end:idx_step]

    descriptions = {"pt": [], "en": []}

    for cluster_ids in ids_to_describe:
      img_paths = [path.join(self.images_dir_path, f"{id}.jpg") for id in cluster_ids]
      for lang in descriptions.keys():
        words = self.llama.common(img_paths, lang=lang)
        descriptions[lang].append(words)

    return descriptions


  def describe_by_siglip2(self, ids_by_distance, num_images=100, max_words=10, word_list_limit=500):
    if self.siglip == None:
      self.siglip = SigLip2()
      self.words = {
        "en": get_caption_words(self.data_file_path, lang="en", categories=["people", "fauna", "flora"])[:word_list_limit],
        "pt": get_caption_words(self.data_file_path, lang="pt", categories=["people", "fauna", "flora"])[:word_list_limit]
      }

    ids_to_avg = ids_by_distance[:, :num_images]
    embeddings_to_avg = np.array([[self.embedding_data[id]["siglip2"] for id in ids] for ids in ids_to_avg])
    embeddings_avg = embeddings_to_avg.mean(axis=1)

    descriptions = {"pt": [], "en": []}

    for cluster_avg in embeddings_avg:
      img_tags_en = self.siglip.zero_shot(cluster_avg, self.words["en"])
      img_tags_pt = self.siglip.zero_shot(cluster_avg, self.words["pt"], prefix="pintura mostrando")
      descriptions["en"].append(img_tags_en[:max_words])
      descriptions["pt"].append(img_tags_pt[:max_words])

    return descriptions
