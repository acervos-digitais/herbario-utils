from gc import collect
from PIL import Image as PImage

from torch import cuda, no_grad, Tensor
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor
from warnings import simplefilter

simplefilter(action="ignore")

class Embedder:
  DEVICE = "cuda" if cuda.is_available() else "cpu"

  @staticmethod
  def scaleMinMax(data):
    dmin = data.min()
    dmax = data.max()
    return (data - dmin) / (dmax - dmin + 1e-10)

  def __init__(self, model_name):
    self.processor = AutoProcessor.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name).to(Embedder.DEVICE)

  def cleanup(self):
    if self.model is not None:
      self.model.to("cpu")
      del self.model
      self.model = None
    self.processor = None
    collect()
    if cuda.is_available():
      cuda.empty_cache()
      cuda.ipc_collect()

  def get_image_embedding(self, img):
    inputs = self.processor(images=img, return_tensors="pt").to(self.model.device)

    with no_grad():
      img_embedding = self.model.get_image_features(**inputs).pooler_output.detach().cpu().squeeze()
      img_embedding = F.normalize(img_embedding, p=2, dim=-1)

    return img_embedding


  def get_text_embedding(self, text, prefix=None):
    text = [text] if type(text) == str else text
    if prefix:
      text = [f"{prefix} {t}" for t in text]
    txt_input = self.processor(text=text, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.model.device)

    with no_grad():
      txt_embedding = self.model.get_text_features(**txt_input).pooler_output.cpu().squeeze()
      txt_embedding = F.normalize(txt_embedding, p=2, dim=-1)

    return txt_embedding


  def get_cosine_similarities(self, img, texts, prefix=None):
    txt_embeddings = texts
    if type(texts[0]) == str:
      txt_embeddings = self.get_text_embedding(texts, prefix=prefix)
    if not isinstance(txt_embeddings, Tensor):
      txt_embeddings = Tensor(txt_embeddings)

    img_embedding = img
    if isinstance(img, PImage.Image):
      img_embedding = self.get_image_embedding(img)
    if not isinstance(img_embedding, Tensor):
      img_embedding = Tensor(img_embedding)

    return (img_embedding @ txt_embeddings.T).cpu().squeeze().numpy()


  def zero_shot(self, img, texts, prefix=None):
    sim_scores = self.get_similarity_scores(img, texts, prefix=prefix)
    text_idxs_by_similarity = (-sim_scores).argsort()

    if type(texts[0]) == str:
      return [texts[idx] for idx in text_idxs_by_similarity]
    else:
      return text_idxs_by_similarity


  def shot_zero(self, embeddings, texts, prefix=None):
    txt_embeddings = texts
    if type(texts[0]) == str:
      txt_embeddings = self.get_text_embedding(texts, prefix=prefix)
    if not isinstance(txt_embeddings, Tensor):
      txt_embeddings = Tensor(txt_embeddings)

    sim_scores = (txt_embeddings @ embeddings.T).cpu().squeeze().numpy()
    return (-sim_scores).argsort()
