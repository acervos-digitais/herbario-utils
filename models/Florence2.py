from torch import no_grad, tensor
from transformers import AutoProcessor, Florence2ForConditionalGeneration
from warnings import simplefilter

from .ObjectDetector import ObjectDetector

simplefilter(action="ignore")

from IPython.display import display

class Florence2(ObjectDetector):
  MODEL_NAME = "florence-community/Florence-2-large-ft"

  OBJECT_MIN_D = 0.01
  OBJECT_MAX_D = 0.60

  @classmethod
  # filter if box "too large" or "too small"
  def size_threshold(cls, box_pct):
    box_width = box_pct[2] - box_pct[0]
    box_height = box_pct[3] - box_pct[1]
    good_min = box_width > cls.OBJECT_MIN_D and box_height > cls.OBJECT_MIN_D
    good_max = box_width < cls.OBJECT_MAX_D and box_height < cls.OBJECT_MAX_D
    return good_min and good_max

  def __init__(self, model=None, all_labels=None):
    super().__init__(model, all_labels)
    model_name = Florence2.MODEL_NAME if model is None else model
    self.processor = AutoProcessor.from_pretrained(model_name)
    self.model = Florence2ForConditionalGeneration.from_pretrained(model_name).to(ObjectDetector.DEVICE)

  def run_object_detection(self, img, model_label):
    prompt = "<OPEN_VOCABULARY_DETECTION>" + model_label

    inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.model.device)
    with no_grad():
      generated_ids = self.model.generate(
        **inputs,
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
      )

    generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = self.processor.post_process_generation(
      generated_text,
      task="<OPEN_VOCABULARY_DETECTION>",
      image_size=img.size
    )

    res = parsed_answer["<OPEN_VOCABULARY_DETECTION>"]

    boxes = tensor(res.get("bboxes", []))
    boxes = [ObjectDetector.px_to_pct(b, img.size) for b in boxes]

    detected_objs = [
      {
        "label": model_label,
        "box": b
      }
      for b in boxes if Florence2.size_threshold(b)
    ]

    return detected_objs
