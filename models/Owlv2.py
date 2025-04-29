import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection, Owlv2VisionModel
from warnings import simplefilter

simplefilter(action="ignore")

class Owlv2:
  OBJ_TARGET_SIZE = torch.Tensor([500, 500])
  MODEL_NAME = "google/owlv2-base-patch16"
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

  @classmethod
  def px_to_pct(cls, box, img_w, img_h):
    scale_factor = torch.tensor([max(img_w, img_h) / img_w , max(img_w, img_h) / img_h])
    img_dims = cls.OBJ_TARGET_SIZE / scale_factor
    return [round(x, 4) for x in (box.cpu().reshape(2, -1) / img_dims).reshape(-1).tolist()]

  @classmethod
  # filter if box "too large" or "too small"
  def threshold(cls, score, label, box, tholds, img_w, img_h):
    box_pct = cls.px_to_pct(box, img_w, img_h)
    box_width = box_pct[2] - box_pct[0]
    box_height = box_pct[3] - box_pct[1]
    good_min = box_width > 0.05 and box_height > 0.05
    good_max = box_width < 0.8 or box_height < 0.8
    return good_min and good_max and score > tholds[label.item()]

  def __init__(self, model=None):
    model_name = Owlv2.MODEL_NAME if model is None else model
    self.processor = Owlv2Processor.from_pretrained(model_name)
    self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(Owlv2.DEVICE)

  def run_object_detection(self, img, labels, tholds):
    input = self.processor(text=labels, images=img, return_tensors="pt").to(Owlv2.DEVICE)
    with torch.no_grad():
      obj_out = self.model(**input)

    res = self.processor.post_process_object_detection(outputs=obj_out, target_sizes=[Owlv2.OBJ_TARGET_SIZE])
    slbs = zip(res[0]["scores"], res[0]["labels"], res[0]["boxes"])
    iw, ih = img.size

    detected_objs = [{"score": s, "label": labels[l.item()], "box": Owlv2.px_to_pct(b, iw, ih)}
                     for s,l,b in slbs if Owlv2.threshold(s, l, b, tholds, iw, ih)]
    return detected_objs

  def top_objects(self, img, labels, tholds):
    detected_objs = self.run_object_detection(img, labels, tholds)
    by_label_score = sorted(detected_objs, key=lambda x: (x["label"], x["score"]))
    unique_label = {o["label"]: o["box"] for o in by_label_score}
    return [{"label": k, "box": v} for k,v in unique_label.items()]

  def all_objects(self, img, labels, tholds):
    detected_objs = self.run_object_detection(img, labels, tholds)
    return [{k: o[k] for k in ["box", "label"]} for o in detected_objs]


class Owlv2Embedding:
  MODEL_NAME = "google/owlv2-base-patch16"
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

  def __init__(self, model=None):
    model_name = Owlv2Embedding.MODEL_NAME if model is None else model
    self.processor = Owlv2Processor.from_pretrained(model_name)
    self.model = Owlv2VisionModel.from_pretrained(model_name).to(Owlv2Embedding.DEVICE)

  def get_embedding(self, img):
    input = self.processor(images=img, return_tensors="pt").to(Owlv2Embedding.DEVICE)

    with torch.no_grad():
      output = self.model(**input)

    my_embedding = output["last_hidden_state"][:, 0, :].detach().squeeze()
    # my_embedding = output["last_hidden_state"][:, 1:, :].mean(dim=1).detach().squeeze()

    return my_embedding
