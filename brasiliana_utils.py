import json
import re
import requests
import urllib.request as request

from io import BytesIO

from PIL import Image as PImage, ImageOps as PImageOps
PImage.MAX_IMAGE_PIXELS = None


class Brasiliana:
  API_URL = "https://brasiliana.museus.gov.br/wp-json"
  TAINACAN_URL = f"{API_URL}/tainacan/v2/items"

  CATEGORIES = {
    "painting": 1076,
    "drawing": 1069,
    "print": 1067,
    "money": 15
  }

  ITEM_DATA_FIELDS = {
    "title": "title",
    "creator": "autor",
    "date": "data-de-producao",
    "museum": "instalacao",
    "categories": "classificacao",
    "depicts": "denominacao",
    "caption": "description",
  }

  FIELDS_TO_TRANSLATE = [
    "classificacao",
    "denominacao",
    "description",
  ]

  @classmethod
  def object_from_json_url(cls, url):
    with request.urlopen(url) as in_file:
      return json.load(in_file)

  @classmethod
  def prep_category_query(cls, category_label):
    if category_label not in cls.CATEGORIES:
      raise Exception("Invalid Category:", category_label)

    return f"{cls.TAINACAN_URL}/?" + \
      "perpage=100&" + \
      "order=ASC&orderby=date&" + \
      "taxquery%5B0%5D%5Btaxonomy%5D=tnc_tax_27&" + \
      f"taxquery%5B0%5D%5Bterms%5D%5B0%5D={cls.CATEGORIES[category_label]}&" + \
      "taxquery%5B0%5D%5Bcompare%5D=IN&" + \
      "exposer=json-flat&" + \
      "paged=1"

  @classmethod
  def run_category_query(cls, category_label):
    return cls.run_query(cls.prep_category_query(category_label))

  @classmethod
  def run_query(cls, url):
    result = cls.object_from_json_url(url)
    items = result["items"]

    for i in items:
      for f in cls.FIELDS_TO_TRANSLATE:
        i["data"][f]["value"] = {
          "pt": i["data"][f]["value"],
          "en": ""
        }

    pagination = result["pagination"]
    if(pagination["current_page"] < pagination["total_page"]):
      print("page", pagination["next_page"])
      nurl = re.sub(r"paged=[0-9]+", f"paged={pagination['next_page']}", url)
      items += cls.run_query(nurl)

    return items

  @classmethod
  def download_image(cls, img_url):
    response = requests.get(img_url)
    response.raise_for_status()
    pimg = PImage.open(BytesIO(response.content)).convert("RGB")
    return PImageOps.exif_transpose(pimg)
