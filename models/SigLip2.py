import numpy as np

from PIL import Image as PImage
from sklearn.metrics.pairwise import cosine_distances

from torch import cuda, no_grad, relu, Tensor
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor
from warnings import simplefilter

simplefilter(action="ignore")

class SigLip2:
  MODEL_NAME = "google/siglip2-giant-opt-patch16-256"
  DEVICE = "cuda" if cuda.is_available() else "cpu"

  @classmethod
  def scaleMinMax(cls, data):
    dmin = data.min()
    dmax = data.max()
    return (data - dmin) / (dmax - dmin + 1e-10)

  def __init__(self, model=None):
    model_name = SigLip2.MODEL_NAME if model is None else model
    self.processor = AutoProcessor.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name).to(SigLip2.DEVICE)

    self.model_grid_size = (
      self.model.config.vision_config.image_size //
      self.model.config.vision_config.patch_size
    )

  def get_image_embedding(self, img):
    input = self.processor(images=img, return_tensors="pt").to(SigLip2.DEVICE)

    with no_grad():
      my_embedding = self.model.get_image_features(**input).pooler_output.detach().cpu().squeeze()

    return my_embedding

  def get_text_embedding(self, text):
    txt_input = self.processor(text=text, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(SigLip2.DEVICE)

    with no_grad():
      txt_embedding = self.model.get_text_features(**txt_input).pooler_output.cpu().squeeze()

    return txt_embedding

  def zero_shot(self, img, tags, prefix="painting with a"):
    texts = [f"{prefix} {t}" for t in tags]

    img_embedding = img
    if isinstance(img, PImage.Image):
      img_embedding = self.get_image_embedding(img).cpu()

    txt_input = self.processor(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(SigLip2.DEVICE)

    with no_grad():
      txt_embedding = self.model.get_text_features(**txt_input).pooler_output.cpu()

    dists = cosine_distances(img_embedding.reshape(1, -1), txt_embedding)

    tag_idxs_by_distance = dists[0].argsort()
    return [tags[idx] for idx in tag_idxs_by_distance]

  def shot_zero(self, embeddings, text):
    text = [text] if type(text) == str else text
    text = [f" {t}" for t in text]
    txt_input = self.processor(text=text, padding="max_length", max_length=64, return_tensors="pt").to(SigLip2.DEVICE)

    with no_grad():
      txt_embedding = self.model.get_text_features(**txt_input).pooler_output.cpu()

    dists = cosine_distances(txt_embedding, embeddings)
    return dists.argsort(axis=1)

  def get_gradient_activation_map(self, img, labels, *, img_idx=0, label_idx=None):
    if isinstance(labels, str):
      labels = [labels]

    if isinstance(labels, list) and isinstance(labels[0], str):
      inputs = self.processor(
        text = labels,
        padding="max_length", max_length=64, truncation=True,
        return_tensors="pt"
      ).to(SigLip2.DEVICE)

      with no_grad():
        text_embs = self.model.text_model(**inputs).pooler_output
        labels = text_embs.detach().cpu().numpy()

    if isinstance(labels, np.ndarray) and len(labels.shape) == 1:
      labels = labels[None, :]

    if (isinstance(labels, np.ndarray) or isinstance(labels, Tensor)) and len(labels.shape) == 2:
      labels = Tensor(labels).to(SigLip2.DEVICE)
      return self.get_gradient_activation_map_from_embeddings(img, labels, img_idx=img_idx, label_idx=label_idx)
    else:
      raise TypeError(f"Expected a 2D np.ndarray or Tensor, got {type(labels)}")

  def get_gradient_activation_map_from_embeddings(self, img, labels, *, img_idx=0, label_idx=None):
    label_idxs = range(len(labels)) if label_idx is None else [label_idx]
    label_activations = []

    inputs = self.processor(
      images=img,
      padding="max_length", max_length=64, truncation=True,
      return_tensors="pt"
    ).to(SigLip2.DEVICE)

    outputs = self.model.vision_model(**inputs)
    img_embedding = F.normalize(outputs.pooler_output[img_idx], p=2, dim=-1)

    patch_features = outputs.last_hidden_state
    patch_features.retain_grad()

    for lidx in label_idxs:
      label_embedding = F.normalize(labels[lidx], p=2, dim=-1)
      similarity_score = (label_embedding * img_embedding).sum()

      self.model.zero_grad()
      similarity_score.backward(retain_graph=True) # need L4 GPU with 24GB (15.5GB)
      patch_grads = patch_features.grad
      patch_weights = patch_grads[img_idx].mean(dim=0)

      cam = (patch_weights * patch_features[img_idx]).sum(dim=-1)
      cam = relu(cam)
      label_activations.append(cam.detach().cpu().numpy())

    cam01 = SigLip2.scaleMinMax(np.array(label_activations).mean(axis=0))
    return cam01.reshape(self.model_grid_size, self.model_grid_size)
