from torch import cuda, no_grad, tensor
from warnings import simplefilter

from .SigLip2 import SigLip2

simplefilter(action="ignore")

class ObjectDetector:
  DEVICE = "cuda" if cuda.is_available() else "cpu"

  OBJECT_MIN_D = 0.05
  OBJECT_MAX_D = 0.8

  @staticmethod
  def find_index(lst, item):
    try:
      return lst.index(item)
    except ValueError:
      return -1

  @staticmethod
  def px_to_pct(box, img_size):
    scale_factor = tensor(img_size)
    return [round(x, 4) for x in (box.cpu().reshape(2, -1) / scale_factor).reshape(-1).tolist()]

  @staticmethod
  def iou(boxA, boxB, return_areas=False):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    intersection = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both rectangles
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - intersection

    # compute the intersection over union:
    # union is sum of both areas minus intersection
    iou = intersection / union if union > 0 else 0

    if return_areas:
      return iou, intersection, areaA, areaB
    else:
      return iou

  @staticmethod
  def remove_duplicate_by_score(detected_objs):
    keep = detected_objs[:1]
    for boxObjA in detected_objs[1:]:
      new_keep = []
      boxA = boxObjA["box"]
      scoreA = boxObjA["score"]
      keepA = True
      for boxObjB in keep:
        boxB = boxObjB["box"]
        scoreB = boxObjB["score"]
        same_box = sum([abs(axy - bxy) for axy, bxy in zip(boxA, boxB)]) < 0.001

        if not same_box:
          new_keep.append(boxObjB)
        elif scoreA < scoreB:
          keepA = False
          new_keep.append(boxObjB)

      if keepA:
        new_keep.append(boxObjA)

      keep = new_keep[:]
    return keep

  @classmethod
  def filter_by_iou(cls, detected_objs, iou_thold=0.8, iou_per_label=False):
    objs_to_filter = detected_objs if iou_per_label else cls.remove_duplicate_by_score(detected_objs)
    by_label = {}
    for obj in objs_to_filter:
      obj_label = obj["label"] if iou_per_label else "all"
      by_label[obj_label] = by_label.get(obj_label, []) + [obj]

    ioud_by_label = {}
    for k, all_boxes in by_label.items():
      keep = all_boxes[:1]
      for boxObjA in all_boxes[1:]:
        new_keep = []
        boxA = boxObjA["box"]
        keepA = True
        for boxObjB in keep:
          boxB = boxObjB["box"]
          iouAB, _, areaA, areaB = cls.iou(boxA, boxB, return_areas=True)

          if iouAB < iou_thold:
            new_keep.append(boxObjB)
          elif areaA < areaB:
            keepA = False
            new_keep.append(boxObjB)

        if keepA:
          new_keep.append(boxObjA)

        keep = new_keep[:]
      ioud_by_label[k] = keep

    return [obj for objs in ioud_by_label.values() for obj in objs]

  @classmethod
  # filter if box "too large" or "too small"
  def size_score_threshold(cls, score, label_idx, box_pct, tholds):
    box_width = box_pct[2] - box_pct[0]
    box_height = box_pct[3] - box_pct[1]
    good_min = box_width > cls.OBJECT_MIN_D and box_height > cls.OBJECT_MIN_D
    good_max = box_width < cls.OBJECT_MAX_D or box_height < cls.OBJECT_MAX_D
    return good_min and good_max and score > tholds[label_idx]

  def __init__(self, model=None, all_labels=None):
    if all_labels:
      self.all_labels = all_labels
      self.siglip = SigLip2()
      # pre-compute embeddings for all output labels
      self.all_labels_embeddings = self.siglip.get_text_embedding(all_labels)

  def alignment_threshold(self, box_pct, img, predicted_label, logits_abs_thold=-2.0, logits_rel_thold=2.0):
    iw, ih = img.size
    x0,y0,x1,y1 = box_pct
    box_width = x1 - x0
    box_height = y1 - y0

    crop_pad = 0.015
    x0,y0 = max(x0-crop_pad, 0), max(y0-crop_pad, 0)
    x1,y1 = min(x1+crop_pad, 1), min(y1+crop_pad, 1)

    oimg = img.crop((x0*iw, y0*ih, x1*iw, y1*ih))
    pfix = "a painting of"
    logits = self.siglip.get_logits(oimg, self.all_labels_embeddings, prefix=pfix)

    predicted_label_idx = self.all_labels.index(predicted_label)
    logit_label_idx = logits.argmax()

    all_logits_ordered = [int(logits[i]*100)/100 for i in (-logits).argsort()]
    logits_margin = all_logits_ordered[0] - all_logits_ordered[1]

    return predicted_label_idx == logit_label_idx and logits.max() > logits_abs_thold and logits_margin > logits_rel_thold

  def run_object_detection(self, img, model_labels, tholds, combined_label=None):
    # Make Labels Great Again
    if type(model_labels) == str and " . " in model_labels:
      labels_str = model_labels
      labels = model_labels.split(" . ")
    elif type(model_labels) == str:
      labels_str = model_labels
      labels = [model_labels]
    elif type(model_labels) == list:
      labels_str = " . ".join(model_labels)
      labels = model_labels

    inputs = self.processor(text=model_labels, images=img, return_tensors="pt").to(ObjectDetector.DEVICE)
    with no_grad():
      outputs = self.model(**inputs)

    res = self.processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=[img.size[::-1]])

    # Make Results Great Again
    res[0]["scores"] = res[0]["scores"].tolist()

    if "text_labels" not in res[0] or res[0]["text_labels"] is None:
      res[0]["text_labels"] = [labels[i] for i in res[0]["labels"]]

    if "labels" not in res[0] or res[0]["labels"] is None or len(res[0]["labels"]) < 1 or type(res[0]["labels"][0]) == str:
      res[0]["labels"] = [ObjectDetector.find_index(labels, l) for l in res[0]["text_labels"]]
    else:
      res[0]["labels"] = res[0]["labels"].tolist()

    res[0]["boxes"] = [ObjectDetector.px_to_pct(b, img.size) for b in res[0]["boxes"]]

    slbs = zip(res[0]["scores"], res[0]["labels"], res[0]["boxes"])

    detected_objs = [
      {
        "score": round(s, 3),
        "label": combined_label if combined_label else labels[l],
        "box": b
      }
      for s,l,b in slbs if self.size_score_threshold(s, l, b, tholds)
    ]
    return detected_objs

  def top_objects(self, img, labels, tholds):
    detected_objs = self.run_object_detection(img, labels, tholds)
    by_label_score = sorted(detected_objs, key=lambda x: (x["label"], x["score"]))
    unique_label = {o["label"]: o for o in by_label_score}
    return list(unique_label.values())

  def all_objects(self, img, labels, tholds):
    detected_objs = self.run_object_detection(img, labels, tholds)
    return detected_objs

  def iou_objects(self, img, labels, tholds, combined_label=None, iou_per_label=True):
    detected_objs = self.run_object_detection(img, labels, tholds, combined_label=combined_label)
    ioud_objs = self.filter_by_iou(detected_objs, iou_thold=0.55, iou_per_label=iou_per_label)
    return ioud_objs
