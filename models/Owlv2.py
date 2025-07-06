import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection, Owlv2Model
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

  @classmethod
  def iou(cls, boxA, boxB, return_areas=False):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both rectangles
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union:
    # union is sum of both areas minus intersection
    iou = intersection / float(areaA + areaB - intersection)

    if return_areas:
      return iou, intersection, areaA, areaB
    else:
      return iou

  @classmethod
  def remove_duplicate_by_score(cls, detected_objs):
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

    detected_objs = [{"score": round(s.item(), 3), "label": labels[l.item()], "box": Owlv2.px_to_pct(b, iw, ih)}
                     for s,l,b in slbs if Owlv2.threshold(s, l, b, tholds, iw, ih)]
    return detected_objs

  def top_objects(self, img, labels, tholds):
    detected_objs = self.run_object_detection(img, labels, tholds)
    by_label_score = sorted(detected_objs, key=lambda x: (x["label"], x["score"]))
    unique_label = {o["label"]: o for o in by_label_score}
    return list(unique_label.values())

  def all_objects(self, img, labels, tholds):
    detected_objs = self.run_object_detection(img, labels, tholds)
    return detected_objs

  def iou_objects(self, img, labels, tholds):
    detected_objs = self.run_object_detection(img, labels, tholds)
    ioud_objs = self.filter_by_iou(detected_objs, iou_thold=0.8)
    return ioud_objs

  def get_objectness_boxes(self, img, topk=8):
    tsize = [img.size[::-1]]
    input = self.processor(images=img, text="", return_tensors="pt").to(Owlv2.DEVICE)
    with torch.no_grad():
      output = self.model(**input)

    objectnesses = output["objectness_logits"].squeeze()
    objectness_idxs = torch.sort(objectnesses)[1][-topk:].tolist()
    pred_boxes = self.processor.post_process_object_detection(outputs=output, target_sizes=tsize, threshold=0)[0]["boxes"]

    crop_boxes = [[int(i) for i in pred_boxes[idx].tolist()] for idx in objectness_idxs]
    return crop_boxes


class Owlv2Embedding:
  MODEL_NAME = "google/owlv2-base-patch16"
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

  def __init__(self, model=None):
    model_name = Owlv2Embedding.MODEL_NAME if model is None else model
    self.processor = Owlv2Processor.from_pretrained(model_name)
    self.model = Owlv2Model.from_pretrained(model_name).to(Owlv2Embedding.DEVICE)

  def get_embedding(self, img):
    input = self.processor(images=img, return_tensors="pt").to(Owlv2Embedding.DEVICE)

    with torch.no_grad():
      output = self.model.get_image_features(**input)

    my_embedding = output.detach().squeeze()

    return my_embedding
