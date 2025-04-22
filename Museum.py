import base64
import json

from os import makedirs, path
from PIL import Image as PImage, ImageOps as PImageOps

from brasiliana_utils import Brasiliana
from wikidata_utils import Wikidata

from dominant_colors import get_dominant_colors
from models.CLIP_embedding import Clip
from models.EnPt import EnPt, PtEn, PartOfSpeech
from models.LlamaVision import LlamaVision
from models.Owlv2 import Owlv2

from params.detect import OBJS_LABELS_IN as OBJS_LABELS, OBJS_THOLDS

class Museum:
  @classmethod
  def prep_dirs(cls, museum_info):
    cls.DIRS = {
      "data": f"./metadata/json/{museum_info['dir']}",
      "imgs": f"../../imgs/{museum_info['dir']}"
    }

    for d in ["captions", "colors", "embeddings", "objects"]:
      cls.DIRS[d] = path.join(cls.DIRS["data"], d)

    cls.IMGS = {}
    for d in ["500", "900", "full"]:
      cls.IMGS[d] = path.join(cls.DIRS["imgs"], d)

    cls.INFO_PATH = path.join(cls.DIRS["data"], f"{museum_info['file']}.json")

  @classmethod
  def read_data(cls):
    museum_data = {}
    if (path.isfile(cls.INFO_PATH)):
      with open(cls.INFO_PATH, "r") as ifp:
        museum_data = json.load(ifp)
    return museum_data

  @classmethod
  def write_data(cls, museum_data):
    with open(cls.INFO_PATH, "w") as ofp:
      json.dump(museum_data, ofp, separators=(",",":"), sort_keys=True, ensure_ascii=False)

  @classmethod
  def get_metadata(cls, museum_info):
    cls.prep_dirs(museum_info)
    makedirs(cls.DIRS["data"], exist_ok=True)

  @classmethod
  def download_images(cls, museum_info):
    cls.prep_dirs(museum_info)

    for d in cls.IMGS.values():
      makedirs(d, exist_ok=True)

    museum_data = cls.read_data()

    qids = sorted(list(museum_data.keys()))
    print(len(qids), "images")

    for cnt,qid in enumerate(qids):
      if cnt % 100 == 0:
        print(cnt)

      img_path_full = path.join(cls.IMGS["full"], f"{qid}.jpg")
      img_path_900 = path.join(cls.IMGS["900"], f"{qid}.jpg")
      img_path_500 = path.join(cls.IMGS["500"], f"{qid}.jpg")

      img_url = museum_data[qid]["image"]

      if (not path.isfile(img_path_full)) or (not path.isfile(img_path_900)) or (not path.isfile(img_path_500)):
        try:
          pimg = cls.download_image(img_url)
        except Exception as e:
          print("\t", qid)
          print("\t", img_url)
          print("\t", e)
          continue

      if (not path.isfile(img_path_full)):
        pimg.thumbnail([4096, 4096])
        pimg.save(img_path_full)

      if (not path.isfile(img_path_900)):
        pimg.thumbnail([900, 900])
        pimg.save(img_path_900)

      if (not path.isfile(img_path_500)):
        pimg.thumbnail([500, 500])
        pimg.save(img_path_500)

  @classmethod
  def get_colors(cls, museum_info):
    cls.prep_dirs(museum_info)
    makedirs(cls.DIRS["colors"], exist_ok=True)

    museum_data = cls.read_data()

    qids = sorted(list(museum_data.keys()))
    print(len(qids), "images")

    for cnt,qid in enumerate(qids):
      if cnt % 100 == 0:
        print(cnt)

      img_path = path.join(cls.IMGS["500"], f"{qid}.jpg")
      color_path = path.join(cls.DIRS["colors"], f"{qid}.json")

      if (not path.isfile(img_path)) or path.isfile(color_path):
        continue

      img = PImage.open(img_path)
      _, rgb_by_hls = get_dominant_colors(img)
      palette = [[int(v) for v in c] for c in rgb_by_hls[:4]]

      color_data = { qid: { "color_palette": palette } }

      with open(color_path, "w", encoding="utf-8") as ofp:
        json.dump(color_data, ofp, sort_keys=True, separators=(",",":"), ensure_ascii=False)

  @classmethod
  def get_embeddings(cls, museum_info):
    cls.prep_dirs(museum_info)
    makedirs(cls.DIRS["embeddings"], exist_ok=True)

    museum_data = cls.read_data()

    qids = sorted(list(museum_data.keys()))
    print(len(qids), "images")

    if not hasattr(cls, "clip"):
      cls.clip = Clip()

    for cnt,qid in enumerate(qids):
      if cnt % 100 == 0:
        print(cnt)

      img_path = path.join(cls.IMGS["500"], f"{qid}.jpg")
      embedding_path = path.join(cls.DIRS["embeddings"], f"{qid}.json")

      if (not path.isfile(img_path)) or path.isfile(embedding_path):
        continue

      img = PImage.open(img_path)
      clip_embedding = [round(v, 8) for v in cls.clip.get_embedding(img).tolist()]

      embedding_data = { qid: { "clip": clip_embedding } }

      with open(embedding_path, "w", encoding="utf-8") as ofp:
        json.dump(embedding_data, ofp, sort_keys=True, separators=(",",":"), ensure_ascii=False)

  @classmethod
  def get_objects(cls, museum_info):
    cls.prep_dirs(museum_info)
    makedirs(cls.DIRS["objects"], exist_ok=True)

    museum_data = cls.read_data()

    qids = sorted(list(museum_data.keys()))
    print(len(qids), "images")

    if not hasattr(cls, "owl"):
      cls.owl = Owlv2("google/owlv2-base-patch16")

    for cnt,qid in enumerate(qids):
      if cnt % 100 == 0:
        print(cnt)

      img_path = path.join(cls.IMGS["900"], f"{qid}.jpg")
      object_path = path.join(cls.DIRS["objects"], f"{qid}.json")

      if (not path.isfile(img_path)) or path.isfile(object_path):
        continue

      image = PImageOps.exif_transpose(PImage.open(img_path).convert("RGB"))

      image_boxes = []
      for labels,tholds in zip(OBJS_LABELS, OBJS_THOLDS):
        obj_boxes = cls.owl.all_objects(image, labels, tholds)
        image_boxes += obj_boxes

      object_data = { qid: { "objects": image_boxes}}

      with open(object_path, "w", encoding="utf-8") as of:
        json.dump(object_data, of, sort_keys=True, separators=(",",":"), ensure_ascii=False)

  @classmethod
  def get_captions(cls, museum_info):
    cls.prep_dirs(museum_info)
    makedirs(cls.DIRS["captions"], exist_ok=True)

    museum_data = cls.read_data()

    qids = sorted(list(museum_data.keys()))
    print(len(qids), "images")

    OLLAMA_URL = "http://127.0.0.1:11434"
    if not hasattr(cls, "llama"):
      cls.llama = LlamaVision()
    if not hasattr(cls, "enpt"):
      cls.enpt = EnPt()

    for cnt,qid in enumerate(qids):
      if cnt % 100 == 0:
        print(cnt)

      img_path = path.join(cls.IMGS["900"], f"{qid}.jpg")
      caption_path = path.join(cls.DIRS["captions"], f"{qid}.json")

      if (not path.isfile(img_path)) or path.isfile(caption_path):
        continue

      with open(img_path, "rb") as ifp:
        img_data = ifp.read()
        img = base64.b64encode(img_data).decode()
        llama_vision_caption_en = cls.llama.caption(img)
        llama_vision_caption_pt = {k:[cls.enpt.translate(w) for w in v] for k,v in llama_vision_caption_en.items()}

        llama_cap = {
          "llama3.2": {
            "en": llama_vision_caption_en,
            "pt": llama_vision_caption_pt
          }
        }

        cap_data = { qid: llama_cap }

        with open(caption_path, "w", encoding="utf-8") as ofp:
          json.dump(cap_data, ofp, sort_keys=True, separators=(",",":"), ensure_ascii=False)

  @classmethod
  def combine_data(cls, museum_info):
    cls.prep_dirs(museum_info)

    museum_data = cls.read_data()

    qids = sorted(list(museum_data.keys()))

    for cnt,qid in enumerate(qids):
      if cnt % 32 == 0:
        print(cnt)

      img_path = path.join(cls.IMGS["500"], f"{qid}.jpg")

      if not path.isfile(img_path):
        print("deleting:", qid)
        del museum_data[qid]
        continue

      for d in ["colors", "embeddings", "objects", "captions"]:
        with open(path.join(cls.DIRS[d], f"{qid}.json"), "r") as ifp:
          data = json.load(ifp)
          museum_data[qid] |= data[qid]

    full_data_path = path.join(cls.DIRS["data"], f"{museum_info['file']}_full.json")
    with open(full_data_path, "w") as ofp:
      json.dump(museum_data, ofp, separators=(",",":"), sort_keys=True, ensure_ascii=False)


class WikidataMuseum(Museum):
  download_image = Wikidata.download_image

  @classmethod
  def get_metadata(cls, museum_info):
    Museum.get_metadata(museum_info)
    museum_data = cls.read_data()

    defval = {"value": "unknown"}

    locations = [museum_info["label"]]
    if museum_info["collection"]:
      locations.append(museum_info["label"] + " collection")

    for category in museum_info["objects"]:
      for location in locations:
        print(location, category)

        cQuery = Wikidata.prep_category_query(category, location)
        cResults = Wikidata.run_query(cQuery)

        for cnt,result in enumerate(cResults):
          if cnt % 100 == 0:
            print(cnt)

          id = result["qid"]["value"]
          cat = result["cat_en"]["value"]

          if id in museum_data:
            cats = set(museum_data[id]["categories"])
            cats.add(cat)
            museum_data[id]["categories"] = list(cats)
            continue

          dResultsEn = Wikidata.run_depicts_query(id, "en")
          dResultsPt = Wikidata.run_depicts_query(id, "pt")

          museum_data[id] = {
            "id": result["qid"]["value"],
            "categories": [cat],
            "depicts": {
              "en": [d["depictsLabel"]["value"] for d in dResultsEn],
              "pt": [d["depictsLabel"]["value"] for d in dResultsPt]
            },
            "title": result["itemLabel"]["value"],
            "date": result.get("date", defval)["value"],
            "creator": result.get("creatorLabel", defval)["value"],
            "image": result["image"]["value"],
            "museum": museum_info["label"]
          }

    cls.write_data(museum_data)


class BrasilianaMuseum(Museum):
  download_image = Brasiliana.download_image

  @classmethod
  def get_metadata(cls, museum_info):
    Museum.get_metadata(museum_info)
    museum_data = cls.read_data()

    if not hasattr(cls, "enpt"):
      cls.enpt = EnPt()
    if not hasattr(cls, "pten"):
      cls.pten = PtEn()
    if not hasattr(cls, "pos"):
      cls.pos = PartOfSpeech()

    for category in museum_info["objects"]:
      print(category)
      qResults = Brasiliana.run_category_query(category)

      for cnt,result in enumerate(qResults):
        if cnt % 100 == 0:
          print(cnt)

        if "http" not in result["document"]["value"]:
          continue

        id = result["id"]

        desc_pt = result["data"]["description"]["value"].replace("&#034;", "")
        desc_en = cls.pten.translate(desc_pt).lower()
        dep_en = cls.pos.get_nouns(desc_en)
        dep_pt = [cls.enpt.translate(t).lower() for t in dep_en]

        if id in museum_data:
          cats = set(museum_data[id]["categories"])
          cats.add(category)
          museum_data[id]["categories"] = list(cats)
          continue

        item_data = {
          "id": result["id"],
          "categories": [category],
          "depicts": {
            "en": dep_en,
            "pt": dep_pt
          },
          "image": result["document"]["value"],
        }

        for k,v in Brasiliana.ITEM_DATA_FIELDS.items():
          item_data[k] = result["data"][v]["value"].replace("&#034;", "")

        museum_data[id] = museum_data.get(id, {}) | item_data

    cls.write_data(museum_data)
