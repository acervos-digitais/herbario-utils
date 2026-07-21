import json
import re
import requests
import urllib.request as request

from io import BytesIO

from PIL import Image as PImage, ImageOps as PImageOps

PImage.MAX_IMAGE_PIXELS = None


class Tainacan:
  API_URL = "https://brasiliana.museus.gov.br/wp-json"
  TAINACAN_URL = f"{API_URL}/tainacan/v2/items"

  USER_AGENT = "Acervos-Digitais/0.1 (https://www.acervosdigitais.fau.usp.br/; acervosdigitais@usp.br)"

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
  }

  @classmethod
  def object_from_json_url(cls, url):
    with request.urlopen(url) as in_file:
      return json.load(in_file)

  @classmethod
  def run_category_query(cls, category_label):
    return cls.run_query(cls.prep_category_query(category_label))

  @classmethod
  def run_query(cls, url):
    result = cls.object_from_json_url(url)
    items = result["items"]

    pagination = result["pagination"]
    if (pagination["current_page"] < pagination["total_page"]) and pagination["next_page"]:
      print("page", pagination["next_page"])
      nurl = re.sub(r"paged=[0-9]+", f"paged={pagination['next_page']}", url)
      items += cls.run_query(nurl)

    return items

  @classmethod
  def download_image(cls, img_url):
    response = requests.get(img_url, headers={"User-Agent": cls.USER_AGENT})
    response.raise_for_status()
    pimg = PImage.open(BytesIO(response.content)).convert("RGB")
    return PImageOps.exif_transpose(pimg)


class Brasiliana(Tainacan):
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
  }

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


class File(Tainacan):
  API_URL = "https://archive.file.org.br/wp-json"
  TAINACAN_URL = f"{API_URL}/tainacan/v2/collection/329/items"

  CATEGORIES = {
    "animation": 90,
    "architecture": 62,
    "algorithmic art": 50,
    "public art": 228,
    "robotic art": 86,
    "synthetic art": 3162,
    "sound art": 170,
    "bio art": 233,
    "cinema": 102,
    "dance": 84,
    "digital photography": 194,
    "fractal art": 199,
    "games": 133,
    "generative art": 209,
    "art installation": 28,
    "internet art": 159,
    "digital language": 55,
    "art mapping": 216,
    "mobile art": 222,
    "performance": 166,
    "digital poetry": 189,
    "award": 42,
    "symposium": 241,
    "software art": 44,
    "artificial life": 40,
    "video art": 183
  }

  ITEM_DATA_FIELDS = {
    "title": "title-5",
    "creator": "nome-artistico-2",
    "country": "pais-2",
    "date": "ano-edicao-4",
    "description": "description-5",
    "museum": "museum",
  }


  @classmethod
  def prep_category_query(cls, category_label):
    if category_label not in cls.CATEGORIES:
      raise Exception("Invalid Category:", category_label)

    return f"{cls.TAINACAN_URL}" + \
      "?perpage=96" + \
      "&order=ASC&orderby=id" + \
      "&taxquery%5B0%5D%5Btaxonomy%5D=tnc_tax_403" + \
      f"&taxquery%5B0%5D%5Bterms%5D%5B0%5D={cls.CATEGORIES[category_label]}" + \
      "&taxquery%5B0%5D%5Bcompare%5D=IN" + \
      "&exposer=json-flat" + \
      "&paged=1"


  @classmethod
  def run_query(cls, url):
    items = Brasiliana.run_query(url)
    for it in items:
      it["id"] = f"F{it['id']}"
      it["data"]["museum"] = { "value": "FILE" }
      country = re.findall(r"\(([A-Z]{2,3})\)", it["data"]["pais-2"]["value"])
      if len(country) > 0 and (len(country[0]) == 2 or len(country[0]) == 3):
        it["data"]["pais-2"]["value"] = country[0]
    return items
