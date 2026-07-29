from torch import no_grad
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from warnings import simplefilter

from .Owlv2 import Owlv2
from params.detect import DinoObjects as DObs

simplefilter(action="ignore")

class Dino(Owlv2):
  MODEL_NAME = "IDEA-Research/grounding-dino-base"
  OBJS_LABELS_IN = [sorted(o.keys()) for o in DObs.OBJECTS]
  OBJS_LABELS_OUT = [[DObs.OBJECT2LABEL.get(l, l) for l in oli] for oli in OBJS_LABELS_IN]
  OBJS_THOLDS = [[DObs.OBJECTS[i][k] for k in oli] for i,oli in enumerate(OBJS_LABELS_IN)]

  def __init__(self, model=None):
    model_name = Dino.MODEL_NAME if model is None else model
    self.processor = AutoProcessor.from_pretrained(model_name)
    self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(Owlv2.DEVICE)

  @classmethod
  def find_index(cls, lst, item):
    try:
      return lst.index(item)
    except ValueError:
      return -1

  def run_object_detection(self, img, labels, tholds):
    labels_str = " . ".join(labels)

    inputs = self.processor(text=labels_str, images=img, return_tensors="pt").to(Owlv2.DEVICE)
    with no_grad():
      outputs = self.model(**inputs)

    res = self.processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=[img.size[::-1]])
    res[0]["scores"] = res[0]["scores"].tolist()
    res[0]["labels"] = [Dino.find_index(labels, l) for l in res[0]["text_labels"]]

    slbs = zip(res[0]["scores"], res[0]["labels"], res[0]["boxes"])
    iw, ih = img.size

    detected_objs = [{"score": round(s, 3), "label": labels[l], "box": Owlv2.px_to_pct(b, iw, ih)}
                     for s,l,b in slbs if Owlv2.threshold(s, l, b, tholds, iw, ih, min_d=0.02)]
    return detected_objs
