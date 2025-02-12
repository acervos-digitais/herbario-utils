import torch

from transformers import CLIPProcessor, CLIPModel

from .EmbeddingModel import EmbeddingModel

class Clip(EmbeddingModel):
  MODEL_NAME = "openai/clip-vit-large-patch14"

  def __init__(self):
    self.processor = CLIPProcessor.from_pretrained(Clip.MODEL_NAME)
    self.model = CLIPModel.from_pretrained(Clip.MODEL_NAME).to(EmbeddingModel.device)

  def get_embedding(self, imgs):
    inputs = self.processor(images=imgs, return_tensors="pt", padding=True).to(Clip.device)

    with torch.no_grad():
      my_embedding = self.model.get_image_features(**inputs).detach().squeeze()

    return my_embedding
