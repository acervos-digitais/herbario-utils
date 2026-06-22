from brasiliana_utils import Brasiliana

class File(Brasiliana):
  TAINACAN_URL = "https://archive.file.org.br/wp-json/tainacan/v2/collection/329/items"

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
    "installation art": 28,
    "internet art": 159,
    "digital language": 55,
    "mapping art": 216,
    "mobile art": 222,
    "performace": 166,
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
    return items
