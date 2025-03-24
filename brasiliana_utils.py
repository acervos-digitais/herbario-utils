import json
import re
import urllib.request as request

from models.EnPt import PtEn

"""
i["id"]
i["url"]
i["document"]["value"]
"""

class Brasiliana:
  API_URL = "https://brasiliana.museus.gov.br/wp-json"
  TAINACAN_URL = f"{API_URL}/tainacan/v2/items"

  # TODO: add categories
  QUERY_PAINTING_URL = f"{TAINACAN_URL}/?" + \
    "perpage=100&" + \
    "order=ASC&orderby=date&" + \
    "taxquery%5B0%5D%5Btaxonomy%5D=tnc_tax_27&" + \
    "taxquery%5B0%5D%5Bterms%5D%5B0%5D=1076&" + \
    "taxquery%5B0%5D%5Bcompare%5D=IN&" + \
    "exposer=json-flat&" + \
    "paged=1"

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
  def run_category_query(cls, category):
    # TODO: add categories
    if category == "painting":
      return cls.run_query(cls.QUERY_PAINTING_URL)

  @classmethod
  def run_query(cls, url):
    # TODO: get json from online
    result = cls.object_from_json_url(url)
    items = result["items"]

    for i in items:
      for f in cls.FIELDS_TO_TRANSLATE:
        i["data"][f]["value"] = {
          "pt": i["data"][f]["value"],
          "en": ""
        }

        if len(i["data"][f]["value"]["pt"]) > 0:
          i["data"][f]["value"]["en"] = PtEn.translate(i["data"][f]["value"]["pt"])

    pagination = result["pagination"]
    if(pagination["current_page"] < pagination["total_page"]):
      print("get page:", pagination["next_page"])
      nurl = re.sub(r"paged=[0-9]+", f"paged={pagination['next_page']}", url)
      items += cls.run_query(nurl)

    return items
