from torch import cuda, no_grad
from transformers import CLIPProcessor, CLIPModel

class Clip:
  MODEL_NAME = "openai/clip-vit-large-patch14"
  DEVICE = "cuda" if cuda.is_available() else "cpu"

  def __init__(self):
    self.processor = CLIPProcessor.from_pretrained(Clip.MODEL_NAME)
    self.model = CLIPModel.from_pretrained(Clip.MODEL_NAME).to(Clip.DEVICE)

  def get_image_embedding(self, img):
    input = self.processor(images=img, return_tensors="pt", padding=True).to(Clip.DEVICE)

    with no_grad():
      my_embedding = self.model.get_image_features(**input).detach().squeeze()

    return my_embedding
