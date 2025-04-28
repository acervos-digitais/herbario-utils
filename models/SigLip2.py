import torch

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
      my_embedding = self.model.get_image_features(**input).detach().squeeze()

    return my_embedding
