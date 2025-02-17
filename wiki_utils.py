import json
import requests

from hashlib import md5
from io import BytesIO

from PIL import Image as PImage, ImageOps as PImageOps
from SPARQLWrapper import SPARQLWrapper

PImage.MAX_IMAGE_PIXELS = None


class Wikidata:
  QUERY_URL = "https://query.wikidata.org/sparql"
  CLAIM_URL = "https://www.wikidata.org/w/api.php?action=wbgetclaims&format=json&property=P18&entity="
  MEDIA_URL = "https://upload.wikimedia.org/wikipedia/commons"

  USER_AGENT = "Acervos-Digitais/0.1 (https://www.acervosdigitais.fau.usp.br/; acervosdigitais@usp.br)"

  QCODES = {
    "photograph": "Q125191",
    "floor plan": "Q18965",
    "postcard": "Q192425",
    "map": "Q4006",
    "painting": "Q3305213",
    "print": "Q11060274",
    "topographic map": "Q216526",
    "printed matter": "Q1261026",
    "ornament": "Q335261",
    "negative": "Q595597",
    "toy": "Q11422",
    "plate": "Q57216",
    "doll": "Q168658",
    "doll clothes": "Q44201312",
    "vase": "Q191851",
    "towel": "Q131696",
    "saucer": "Q1422576",
    "equipment": "Q10273457",
    "photograph album": "Q488053",
    "tin": "Q15706035",
    "teacup": "Q81707",
    "pillowcase": "Q1094401",
    "drawing": "Q93184",
    "light fixture": "Q815738",
    "furniture": "Q14745",
    "lantern": "Q862454",
    "teapot": "Q245005",
    "chair": "Q15026",
    "illustration": "Q178659",
    "product packaging": "Q207822",
    "sculpture": "Q860861",
    "statue": "Q179700",
  }

  @classmethod
  def prep_category_query(cls, category_label):
    return f"""
      SELECT DISTINCT ?item ?itemLabel ?qid ?image ?creatorLabel ?date ?cat_en ?cat_pt WHERE {{
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
      ?item wdt:P276 wd:Q371803.
      ?item wdt:P18 ?image.

      ?item wdt:P31 wd:{cls.QCODES[category_label]}.

      BIND(STRAFTER(STR(?item), STR(wd:)) AS ?qid).

      BIND(wd:{cls.QCODES[category_label]} AS ?category).
      ?category rdfs:label ?cat_en filter (lang(?cat_en) = "en").
      ?category rdfs:label ?cat_pt filter (lang(?cat_pt) = "pt").

      OPTIONAL {{ ?item wdt:P170 ?creator. }}
      OPTIONAL {{ ?item wdt:P571 ?date. }}
    }}
    """

  @classmethod
  def prep_depicts_query(cls, qid, lang="en"):
    return f"""
      SELECT DISTINCT ?qid ?depictsLabel WHERE {{
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}". }}
      BIND(wd:{qid} AS ?item).
      ?item wdt:P180 ?depicts .
      OPTIONAL {{ ?depicts rdfs:label ?depictsLabel FILTER (lang(?depictsLabel) = "{lang}") }}
    }}
    """

  @classmethod
  def run_query(cls, query, endpoint_url=None):
    endpoint_url = cls.QUERY_URL if endpoint_url == None else endpoint_url
    sparql = SPARQLWrapper(endpoint_url, agent=cls.USER_AGENT)
    sparql.setQuery(query)
    sparql.setReturnFormat("json")
    return sparql.query().convert()["results"]["bindings"]

  @classmethod
  def run_category_query(cls, object_label):
    return cls.run_query(cls.prep_category_query(object_label))

  @classmethod
  def run_depicts_query(cls, qid, lang="en"):
    return cls.run_query(cls.prep_depicts_query(qid, lang))

  @classmethod
  def qid_to_img_url(cls, qid):
    response = requests.get(f"{cls.CLAIM_URL}{qid}")
    response.raise_for_status()
    info = json.loads(response.content)
    fname = info["claims"]["P18"][0]["mainsnak"]["datavalue"]["value"].replace(" ", "_")
    fname_md5 = md5(fname.encode())
    md5_hex = fname_md5.hexdigest()
    img_url = f"{cls.MEDIA_URL}/{md5_hex[0]}/{md5_hex[:2]}/{fname}"
    return img_url

  @classmethod
  def download_image(cls, img_src):
    if img_src[0] == "Q" or img_src[0] == "q":
      img_url = cls.qid_to_img_url(img_src)
    elif img_src[:4] == "http":
      img_url = img_src
    else:
      print("ERROR. Check image source")

    response = requests.get(img_url, headers={"User-Agent": cls.USER_AGENT})
    response.raise_for_status()
    pimg = PImage.open(BytesIO(response.content)).convert("RGB")
    return PImageOps.exif_transpose(pimg)
