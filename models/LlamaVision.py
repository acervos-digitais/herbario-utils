import base64
import json

from ollama import Client
from pydantic import BaseModel, conlist

class Caption(BaseModel):
  people: list[str]
  environment: list[str]
  nature: list[str]
  animals: list[str]
  plants: list[str]
  fauna: list[str]
  flora: list[str]
  objects: list[str]
  description: str

class CommonNouns(BaseModel):
  generic_nouns: conlist(str, min_length=2, max_length=4)
  specific_nouns: conlist(str, min_length=2, max_length=4)

class CommonDescriptions(BaseModel):
  en: CommonNouns
  pt: CommonNouns

class LlamaVision:
  def __init__(self, url="http://127.0.0.1:11434"):
    self.client = Client(host=url)

  @classmethod
  def img_to_b64(cls, img_path):
    with open(img_path, "rb") as ifp:
      img_data = ifp.read()
      return base64.b64encode(img_data).decode()

  @classmethod
  def tob64(cls, path_or_paths):
    if type(path_or_paths) == str:
      return LlamaVision.img_to_b64(path_or_paths)
    elif type(path_or_paths) == list:
      return [LlamaVision.img_to_b64(p) for p in path_or_paths]

  def caption(self, img_path, model="gemma3:4b"):
    img = LlamaVision.tob64(img_path)

    response = self.client.chat(
      model=model,
      format=Caption.model_json_schema(),
      options={"temperature": 0},
      messages=[{
          "role": "user",
          # "content": "Analyze this image and describe it using nouns related to people, environment, nature, animals, plants, fauna, flora and objects.",
          # "content": "Describe this image using words related to people, environment, nature, animals, plants, fauna, flora and objects.",
          # "content": "Describe this image using single nouns related to people, environment, nature, animals, plants, fauna, flora and objects.",
          "content": "Describe this image using single nouns related to people, environment, nature, animals, plants, fauna, flora and objects. And then describe the image in a direct, concise, single sentence using no more than 32 words related to people, environment, nature, animals, plants, fauna, flora and objects.",
          "images": [img],
      }]
    )

    # get object
    res_obj = json.loads(response["message"]["content"])

    # separate unstructured description
    description = res_obj.pop("description", "")

    # turn non-empty lists into sets
    res_obj_ne = {k:set(v) for k,v in res_obj.items() if len(v) > 0 and v[0].lower() != "none"}

    # remove redundant fauna and flora from nature, animals and plants.
    #   needed extra categories to get more words in the lists.
    #   for example: grass only shows up in flora if we ask for plants.
    for k1 in ["nature", "animals", "plants"]:
      for k2 in ["fauna", "flora"]:
        if k1 in res_obj_ne and k2 in res_obj_ne:
          res_obj_ne[k1] -= res_obj_ne[k2]
      if k1 in res_obj_ne and len(res_obj_ne[k1]) < 1:
        del res_obj_ne[k1]

    # make set of all words
    word_set = set()
    word_set = word_set.union(*res_obj_ne.values())

    # turn sets back into lists
    res_obj_ne = {k:list(v) for k,v in res_obj_ne.items() if len(v) > 0}

    # add words list and unstructured description to result
    res_obj_ne["all"] = sorted(list(word_set))
    res_obj_ne["unstructured"] = [description]

    return res_obj_ne

  def common(self, img_paths):
    imgs = LlamaVision.tob64(img_paths)

    response = self.client.chat(
      model="gemma3:4b",
      format=CommonDescriptions.model_json_schema(),
      options={"temperature": 0},
      messages=[{
        "role": "user",
        # "content": f"Using few words, what do these paintings have in common? Give a generic description about the style of the paintings using 2 or 3 words and a more specific description about the content of the painting using 2 or 3 words. Be objective. Avoid hyperbole or emotional terms. Give descriptions in {language}.",
        # "content": f"Using few words, what do these paintings have in common? Give a generic description using 2 or 3 words and a more specific description using 2 or 3 words. Be objective. Avoid hyperbole or emotional terms. Give descriptions in {language}.",
        "content": f"Give separate descriptions in portuguese and english. Using few words, what do these paintings have in common? Give a generic description using 2 or 3 words and a more specific description using 2 or 3 words. Be objective. Avoid hyperbole or emotional terms. Give separate descriptions in portuguese and english.",
        "images": imgs,
      }]
    )

    # get object
    res_obj = json.loads(response["message"]["content"])

    # combine generic and specific descriptions
    for lang,groups in res_obj.items():
      res_obj[lang] = [d.lower() for descriptions in groups.values() for d in descriptions]

    return res_obj
