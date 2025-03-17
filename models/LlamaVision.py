import json

from ollama import Client
from pydantic import BaseModel

class Caption(BaseModel):
  people: list[str]
  environment: list[str]
  nature: list[str]
  animals: list[str]
  plants: list[str]
  fauna: list[str]
  flora: list[str]
  objects: list[str]


class LlamaVision():
  def __init__(self, url="http://127.0.0.1:11434"):
    self.client = Client(host=url)

  def caption(self, img):
    response = self.client.chat(
      model="llama3.2-vision",
      format=Caption.model_json_schema(),
      options={"temperature": 0},
      messages=[{
          "role": "user",
          # 'content': 'Analyze this image and describe it using nouns related to people, environment, nature, animals, plants, fauna, flora and objects.',
          # 'content': 'Describe this image using words related to people, environment, nature, animals, plants, fauna, flora and objects.',
          'content': 'Describe this image using single nouns related to people, environment, nature, animals, plants, fauna, flora and objects.',
          'images': [img],
      }]
    )

    res_obj = json.loads(response["message"]["content"])
    res_obj_ne = {k:set(v) for k,v in res_obj.items() if len(v) > 0}

    for k1 in ["nature", "animals", "plants"]:
      for k2 in ["fauna", "flora"]:
        if k1 in res_obj_ne and k2 in res_obj_ne:
          res_obj_ne[k1] -= res_obj_ne[k2]
      # if k1 in res_obj_ne and len(res_obj_ne[k1]) < 1:
        # del res_obj_ne[k1]

    word_set = set()
    word_set = word_set.union(*res_obj_ne.values())

    res_obj_ne = {k:list(v) for k,v in res_obj_ne.items() if len(v) > 0}

    # for words in res_obj_ne.values():
    # word_set += words

    res_obj_ne["all"] = sorted(list(word_set))

    return res_obj_ne
