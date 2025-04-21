import json

from os import makedirs, path

from brasiliana_utils import Brasiliana
from wikidata_utils import Wikidata

from models.EnPt import PtEn

class Museum:
  @classmethod
  def prep_dirs(cls, museum_info):
    Museum.MUSEUM_DATA_DIR = f"./metadata/json/{museum_info['dir']}"
    Museum.MUSEUM_INFO_PATH = path.join(Museum.MUSEUM_DATA_DIR, f"{museum_info['file']}.json")
    makedirs(Museum.MUSEUM_DATA_DIR, exist_ok=True)

  @classmethod
  def read_data(cls):
    museum_data = {}
    if (path.isfile(cls.MUSEUM_INFO_PATH)):
      with open(cls.MUSEUM_INFO_PATH, "r") as ifp:
        museum_data = json.load(ifp)
    return museum_data

  @classmethod
  def write_data(cls, museum_data):
    with open(cls.MUSEUM_INFO_PATH, "w") as ofp:
      json.dump(museum_data, ofp, separators=(',',':'), sort_keys=True, ensure_ascii=False)

  @classmethod
  def download_images(cls, museum_info):
    print("images")
    IMG_DIR = f"../../imgs/{museum_info['dir']}"

    IMG_DIR_FULL = path.join(IMG_DIR, "full")
    IMG_DIR_900 = path.join(IMG_DIR, "900")
    IMG_DIR_500 = path.join(IMG_DIR, "500")

    makedirs(IMG_DIR_FULL, exist_ok=True)
    makedirs(IMG_DIR_900, exist_ok=True)
    makedirs(IMG_DIR_500, exist_ok=True)

    museum_data = cls.read_data()

    for cnt, (qid, info) in enumerate(museum_data.items()):
      if cnt % 100 == 0:
        print(cnt)

      img_path_full = path.join(IMG_DIR_FULL, f"{qid}.jpg")
      img_path_900 = path.join(IMG_DIR_900, f"{qid}.jpg")
      img_path_500 = path.join(IMG_DIR_500, f"{qid}.jpg")
      img_url = info["image"]

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

class WikidataMuseum(Museum):
  download_image = Wikidata.download_image

  @classmethod
  def get_metadata(cls, museum_info):
    cls.prep_dirs(museum_info)
    museum_data = cls.read_data()

    defval = {"value": "unknown"}

    locations = [museum_info['label']]
    if museum_info["collection"]:
      locations.append(f"{museum_info['label']} collection")

    for category in museum_info["objects"]:
      for location in locations:
        print(location, category)

        cQuery = Wikidata.prep_category_query(category, location)
        cResults = Wikidata.run_query(cQuery)

        for cnt,result in enumerate(cResults):
          if cnt % 100 == 0:
            print(cnt)

          id = result["qid"]["value"]

          cat = {
            "en": result["cat_en"]["value"],
            "pt": result["cat_pt"]["value"]
          }

          if id in museum_data:
            for l in ["en", "pt"]:
              mcategories = set(museum_data[id]["categories"][l])
              mcategories.add(cat[l])
              museum_data[id]["categories"][l] = list(mcategories)
            continue

          dResultsEn = Wikidata.run_depicts_query(id, "en")
          dResultsPt = Wikidata.run_depicts_query(id, "pt")

          museum_data[id] = {
            "id": result["qid"]["value"],
            "categories": {
              "en": [cat["en"]],
              "pt": [cat["pt"]]
            },
            "depicts": {
              "en": [d["depictsLabel"]["value"] for d in dResultsEn],
              "pt":[d["depictsLabel"]["value"] for d in dResultsPt]
            },
            "title": result["itemLabel"]["value"],
            "date": result.get("date", defval)["value"],
            "creator": result.get("creatorLabel", defval)["value"],
            "image": result["image"]["value"]
          }

    cls.write_data(museum_data)

class BrasilianaMuseum(Museum):
  download_image = Brasiliana.download_image

  @classmethod
  def get_metadata(cls, museum_info):
    cls.prep_dirs(museum_info)
    museum_data = cls.read_data()

    for category in museum_info["objects"]:
      print(category)
      qResults = Brasiliana.run_category_query(category)
      for cnt,result in enumerate(qResults):
        if cnt % 100 == 0:
          print(cnt)

        if "http" not in result["document"]["value"]:
          continue

        id = result["id"]

        item_data = {
          "id": result["id"],
          "image": result["document"]["value"]
        }

        for k,v in Brasiliana.ITEM_DATA_FIELDS.items():
          item_data[k] = result["data"][v]["value"]
          if v in Brasiliana.FIELDS_TO_TRANSLATE:
            if len(item_data[k]["pt"]) > 0:
              item_data[k]["en"] = PtEn.translate(item_data[k]["pt"])

        museum_data[id] = museum_data.get(id, {}) | item_data

    cls.write_data(museum_data)
