from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from warnings import simplefilter

from .ObjectDetector import ObjectDetector

simplefilter(action="ignore")

class Dino(ObjectDetector):
  MODEL_NAME = "IDEA-Research/grounding-dino-base"

  OBJECT_MIN_D = 0.02
  OBJECT_MAX_D = 0.50

  @classmethod
  def size_score_threshold(cls, score, label_idx, box_pct, tholds):
    box_width = box_pct[2] - box_pct[0]
    box_height = box_pct[3] - box_pct[1]
    good_max = box_width < cls.OBJECT_MAX_D and box_height < cls.OBJECT_MAX_D
    super_result = super().size_score_threshold(score, label_idx, box_pct, tholds)
    return good_max and super_result

  def __init__(self, model=None, all_labels=None):
    super().__init__(model, all_labels)
    model_name = Dino.MODEL_NAME if model is None else model
    self.processor = AutoProcessor.from_pretrained(model_name)
    self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(ObjectDetector.DEVICE)

  def run_object_detection(self, img, labels, tholds, combined_label):
    labels_str = " . ".join(labels)
    detected_objs = super().run_object_detection(img, labels_str, tholds, combined_label)

    detected_objs_aligned = [
      slb for slb in detected_objs if self.alignment_threshold(slb["box"], img, combined_label)
    ]

    return detected_objs_aligned
