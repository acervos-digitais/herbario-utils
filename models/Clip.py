from .Embedder import Embedder

class Clip(Embedder):
  MODEL_NAME = "openai/clip-vit-large-patch14"

  def __init__(self, model=None):
    model_name = Clip.MODEL_NAME if model is None else model
    super().__init__(model_name)

  def get_similarity_scores(self, img, texts, prefix=None):
    cos_sim = self.get_cosine_similarities(img, texts, prefix)
    return cos_sim.squeeze()
