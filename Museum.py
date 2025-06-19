import csv
import json

from os import listdir, makedirs, path
from PIL import Image as PImage, ImageOps as PImageOps

from brasiliana_utils import Brasiliana
from wikidata_utils import Wikidata

from date_utils import get_year
from dominant_colors import get_dominant_colors

from models.CLIP import Clip
from models.EnPt import EnPt, PtEn, PartOfSpeech
from models.LlamaVision import LlamaVision
from models.Owlv2 import Owlv2
from models.SigLip2 import SigLip2

from params.detect import OBJS_LABELS_IN as OBJS_LABELS, OBJS_THOLDS

class Museum:
  @classmethod
  def prep_dirs(cls, museum_info):
    cls.DIRS = {
      "data": path.join("./metadata/json", museum_info["dir"]),
      "imgs": f"../../imgs/arts"
    }

    for d in ["captions", "colors", "embeddings", "objects"]:
      cls.DIRS[d] = path.join(cls.DIRS["data"], d)

    cls.IMGS = {}
    for d in ["500", "900", "full"]:
      cls.IMGS[d] = path.join(cls.DIRS["imgs"], d)

    cls.INFO_PATH = path.join(cls.DIRS["data"], museum_info["file"] + "_metadata.json")

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
  def get_embeddings(cls, museum_info, model="clip"):
    cls.prep_dirs(museum_info)
    makedirs(cls.DIRS["embeddings"], exist_ok=True)

    museum_data = cls.read_data()

    qids = sorted(list(museum_data.keys()))
    print(len(qids), "images")

    if not hasattr(cls, "model"):
      if model == "clip":
        cls.model = Clip()
      elif model == "siglip2":
        cls.model = SigLip2()

    for cnt,qid in enumerate(qids):
      if cnt % 100 == 0:
        print(cnt)

      img_path = path.join(cls.IMGS["500"], f"{qid}.jpg")
      embedding_path = path.join(cls.DIRS["embeddings"], f"{qid}.json")

      if (not path.isfile(img_path)):
        continue

      embedding_data = { qid: {} }
      if path.isfile(embedding_path):
        with open(embedding_path, "r") as ifp:
          embedding_data = json.load(ifp)

      if model in embedding_data[qid]:
        continue

      img = PImage.open(img_path)
      img_embedding = [round(v, 8) for v in cls.model.get_embedding(img).tolist()]

      embedding_data[qid][model] = img_embedding

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
  def get_captions(cls, museum_info, model="gemma3:4b"):
    cls.prep_dirs(museum_info)
    makedirs(cls.DIRS["captions"], exist_ok=True)

    museum_data = cls.read_data()

    qids = sorted(list(museum_data.keys()))
    print(len(qids), "images")

    if not hasattr(cls, "llama"):
      cls.llama = LlamaVision()
    if not hasattr(cls, "enpt"):
      cls.enpt = EnPt()

    for cnt,qid in enumerate(qids):
      if cnt % 100 == 0:
        print(cnt)

      img_path = path.join(cls.IMGS["900"], f"{qid}.jpg")
      caption_path = path.join(cls.DIRS["captions"], f"{qid}.json")

      if (not path.isfile(img_path)):
        continue

      cap_data = { qid: {} }
      if path.isfile(caption_path):
        with open(caption_path, "r") as ifp:
          cap_data = json.load(ifp)

      if model in cap_data[qid]:
        continue

      caption_en = cls.llama.caption(img_path, model=model)
      caption_pt = {k:[cls.enpt.translate(w).lower() for w in v] for k,v in caption_en.items()}

      model_name = model.split(":")[0].split("-")[0]
      model_cap = {
        model_name: {
          "en": caption_en,
          "pt": caption_pt
        }
      }

      cap_data = { qid: model_cap }

      with open(caption_path, "w", encoding="utf-8") as ofp:
        json.dump(cap_data, ofp, sort_keys=True, separators=(",",":"), ensure_ascii=False)

  @classmethod
  def combine_data(cls, museum_info):
    cls.prep_dirs(museum_info)

    museum_data = cls.read_data()
    embed_data = {}

    qids = sorted(list(museum_data.keys()))

    for cnt,qid in enumerate(qids):
      if cnt % 32 == 0:
        print(cnt)

      img_path = path.join(cls.IMGS["500"], f"{qid}.jpg")

      if not path.isfile(img_path):
        print("deleting:", qid)
        del museum_data[qid]
        continue

      for d in ["colors", "objects"]:
        with open(path.join(cls.DIRS[d], f"{qid}.json"), "r") as ifp:
          data = json.load(ifp)
          museum_data[qid] |= data[qid]

      with open(path.join(cls.DIRS["captions"], f"{qid}.json"), "r") as ifp:
        data = json.load(ifp)
        museum_data[qid]["captions"] = data[qid]

      with open(path.join(cls.DIRS["embeddings"], f"{qid}.json"), "r") as ifp:
        data = json.load(ifp)
        embed_data[qid] = data[qid]

    processed_path = path.join(cls.DIRS["data"], museum_info["file"] + "_processed.json")
    embed_path = path.join(cls.DIRS["data"], museum_info["file"] + "_embeddings.json")

    with open(processed_path, "w") as ofp:
      json.dump(museum_data, ofp, separators=(",",":"), sort_keys=True, ensure_ascii=False)

    with open(embed_path, "w") as ofp:
      json.dump(embed_data, ofp, separators=(",",":"), sort_keys=True, ensure_ascii=False)

    # for k in museum_data.keys():
    #   museum_data[k]["embeddings"] = embed_data[k]

    # full_data_path = path.join(cls.DIRS["data"], museum_info["file"] + "_full.json")
    # with open(full_data_path, "w") as ofp:
    #   json.dump(museum_data, ofp, separators=(",",":"), sort_keys=True, ensure_ascii=False)

  @classmethod
  def combine_museums(cls, all_museums, out_dir, out_prefix, data_dirs):
    out_file_template = path.join(out_dir, out_prefix + "_XTYPEX.json")

    for out_type in data_dirs:
      output_file_path = out_file_template.replace("XTYPEX", out_type)
      all_data = {}

      for name,info in all_museums.items():
        Museum.prep_dirs(info)

        with open(Museum.INFO_PATH.replace("_metadata.json", f"_{out_type}.json"), "r") as ifp:
          museum_data = json.load(ifp)

        print("reading:", name, len(museum_data))

        repeat_keys = [k for k in museum_data.keys() if k in all_data]
        print("repeat keys:", repeat_keys)

        all_data |= museum_data

      print("writing", len(all_data))

      with open(output_file_path, "w") as ofp:
        json.dump(all_data, ofp, separators=(",",":"), sort_keys=True, ensure_ascii=False)


  @classmethod
  def export_object_crops(cls, museum_info):
    cls.prep_dirs(museum_info)
    img_path_crops = path.join(cls.DIRS["imgs"], "crops")
    makedirs(img_path_crops, exist_ok=True)

    obj_files = sorted([f for f in listdir(cls.DIRS["objects"]) if f.endswith(".json")])
    for fname in obj_files:
      qid = fname.replace(".json", "")
      with open(path.join(cls.DIRS["objects"], fname), "r") as inp:
        iboxes = json.load(inp)[qid]["objects"]

      if len(iboxes) < 1:
        continue

      image_file_path = path.join(cls.IMGS["full"], fname.replace(".json", ".jpg"))
      image = PImageOps.exif_transpose(PImage.open(image_file_path).convert("RGB"))
      iw,ih = image.size

      for bidx,box in enumerate(iboxes):
        idx_str = f"00000{bidx}"[-4:]
        bipath = path.join(img_path_crops, f"{qid}_{idx_str}.jpg")

        x0,y0,x1,y1 = box["box"]
        bimg = image.crop((int(x0*iw), int(y0*ih), int(x1*iw), int(y1*ih)))
        bimg.save(bipath)


class WikidataMuseum(Museum):
  download_image = Wikidata.download_image

  @classmethod
  def get_metadata(cls, museum_info):
    Museum.get_metadata(museum_info)
    museum_data = cls.read_data()

    defval = {"value": "unknown"}

    locations = [museum_info["file"]]
    if museum_info["collection"]:
      locations.append(museum_info["file"] + " collection")

    for category in museum_info["objects"]:
      for location in locations:
        print(location, category)

        cResults = Wikidata.run_category_query(category, location)

        for cnt,result in enumerate(cResults):
          if cnt % 100 == 0:
            print(cnt)

          id = result["qid"]["value"]
          defurlval = {"value": f"https://www.wikidata.org/wiki/{id}"}

          if id in museum_data:
            cats = set(museum_data[id]["categories"])
            cats.add(category)
            museum_data[id]["categories"] = list(cats)
            continue

          depicts = Wikidata.run_depicts_query(id)

          museum_data[id] = {
            "id": result["qid"]["value"],
            "categories": [category],
            "depicts": {
              "en": [d["depicts_en"]["value"] for d in depicts],
              "pt": [d["depicts_pt"]["value"] for d in depicts]
            },
            "title": result["label_pt"]["value"],
            "date": result.get("date", defval)["value"],
            "creator": result.get("creatorLabel", defval)["value"],
            "image": result["image"]["value"],
            "museum": museum_info["label"],
            "url": result.get("article", defurlval)["value"],
          }
          museum_data[id]["year"] = get_year(str(museum_data[id]["date"]))

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

        depicts = { "en": [], "pt": [] }

        desc_pt = result["data"]["description"]["value"].replace("&#034;", "")
        if desc_pt != "" and len(desc_pt) > 2:
          desc_en = cls.pten.translate(desc_pt).lower()
          depicts["en"] = cls.pos.get_nouns(desc_en)
          depicts["pt"] = [cls.enpt.translate(t).lower() for t in depicts["en"]]

        if id in museum_data:
          cats = set(museum_data[id]["categories"])
          cats.add(category)
          museum_data[id]["categories"] = list(cats)
          continue

        item_data = {
          "id": result["id"],
          "categories": [category],
          "depicts": depicts,
          "image": result["document"]["value"],
          "url": result["url"]
        }

        for k,v in Brasiliana.ITEM_DATA_FIELDS.items():
          item_data[k] = result["data"][v]["value"].replace("&#034;", "")

        museum_data[id] = museum_data.get(id, {}) | item_data
        museum_data[id]["year"] = get_year(str(museum_data[id]["date"]))

    cls.write_data(museum_data)

class MacUspMuseum(Museum):
  download_image = Wikidata.download_image

  @classmethod
  def get_metadata(cls, museum_info):
    MACUSP_CATEGORIES = {
      "painting": "pintura",
      "drawing": "desenho",
    }

    Museum.get_metadata(museum_info)
    museum_data = cls.read_data()

    with open(museum_info["path"], "r") as file:
      reader = csv.DictReader(file)

      for row in reader:
        id = "MU" + str(row["id"])

        if row["image"] == "":
          continue

        categories = []
        for cat in museum_info["objects"]:
          if MACUSP_CATEGORIES[cat] in row["category"].lower():
            categories.append(cat)

        if len(categories) < 1:
          continue

        museum_data[id] = {
          "id": id,
          "categories": categories,
          "depicts": { "en": [], "pt": [] },
          "title": row["title"],
          "date": str(row["date"]).replace(";", ""),
          "creator": row["artist"],
          "image": row["image"],
          "museum": museum_info["label"],
          "url": row["url"],
        }
        museum_data[id]["year"] = get_year(str(museum_data[id]["date"]))

    cls.write_data(museum_data)
