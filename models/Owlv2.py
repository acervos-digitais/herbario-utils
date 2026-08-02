from torch import cuda, no_grad, sort
from transformers import Owlv2Processor, Owlv2ForObjectDetection, Owlv2Model
from warnings import simplefilter

from .ObjectDetector import ObjectDetector

simplefilter(action="ignore")

class Owlv2(ObjectDetector):
  MODEL_NAME = "google/owlv2-base-patch16"

  OBJECT_MIN_D = 0.05
  OBJECT_MAX_D = 0.80

  def __init__(self, model=None, all_labels=None):
    super().__init__(model, all_labels)
    model_name = Owlv2.MODEL_NAME if model is None else model
    self.processor = Owlv2Processor.from_pretrained(model_name)
    self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(ObjectDetector.DEVICE)

  def get_objectness_boxes(self, img, topk=8):
    inputs = self.processor(images=img, text="", return_tensors="pt").to(self.model.device)
    with no_grad():
      outputs = self.model(**inputs)

    objectnesses = outputs["objectness_logits"].squeeze()
    objectness_idxs = sort(objectnesses)[1][-topk:].tolist()
    pred_boxes = self.processor.post_process_grounded_object_detection(outputs=output, target_sizes=[img.size[::-1]], threshold=0)[0]["boxes"]

    crop_boxes = [[int(i) for i in pred_boxes[idx].tolist()] for idx in objectness_idxs]
    return crop_boxes


class Owlv2Embedding:
  MODEL_NAME = "google/owlv2-base-patch16"
  DEVICE = "cuda" if cuda.is_available() else "cpu"

  def __init__(self, model=None):
    model_name = Owlv2Embedding.MODEL_NAME if model is None else model
    self.processor = Owlv2Processor.from_pretrained(model_name)
    self.model = Owlv2Model.from_pretrained(model_name).to(Owlv2Embedding.DEVICE)

  def get_image_embedding(self, img):
    input = self.processor(images=img, return_tensors="pt").to(Owlv2Embedding.DEVICE)

    with no_grad():
      output = self.model.get_image_features(**input)

    my_embedding = output.detach().squeeze()

    return my_embedding
