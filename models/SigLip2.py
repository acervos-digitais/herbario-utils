import PIL.Image as PImage
import torch

from sklearn.metrics.pairwise import cosine_distances

from transformers import AutoModel, AutoProcessor
from warnings import simplefilter

simplefilter(action="ignore")

class SigLip2:
  MODEL_NAME = "google/siglip2-giant-opt-patch16-256"
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

  def __init__(self, model=None):
    model_name = SigLip2.MODEL_NAME if model is None else model
    self.processor = AutoProcessor.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name, device_map="auto").to(SigLip2.DEVICE)

  def get_embedding(self, img):
    input = self.processor(images=img, return_tensors="pt").to(SigLip2.DEVICE)

    with torch.no_grad():
      my_embedding = self.model.get_image_features(**input).detach().cpu().squeeze()

    return my_embedding

  def zero_shot(self, img, tags, prefix="painting with a"):
    texts = [f"{prefix} {t}" for t in tags]

    img_embedding = img
    if isinstance(img, PImage.Image):
      img_embedding = self.get_embedding(img).cpu()

    txt_input = self.processor(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(SigLip2.DEVICE)

    with torch.no_grad():
      txt_embedding = self.model.get_text_features(**txt_input).cpu()

    dists = cosine_distances(img_embedding.reshape(1, -1), txt_embedding)

    tag_idxs_by_distance = dists[0].argsort()
    return [tags[idx] for idx in tag_idxs_by_distance]

  def shot_zero(self, embeddings, text):
    text = [text] if type(text) == str else text
    text = [f" {t}" for t in text]
    txt_input = self.processor(text=text, padding="max_length", max_length=64, return_tensors="pt").to(SigLip2.DEVICE)

    with torch.no_grad():
      txt_embedding = self.model.get_text_features(**txt_input).cpu()

    dists = cosine_distances(txt_embedding, embeddings)
    return dists.argsort(axis=1)
