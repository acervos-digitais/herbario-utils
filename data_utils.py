import json

STOP_WORDS = {
  "en": ["form", "forms", "subject", "subjects", "figure", "figures", "observer", "observers", "viewer", "viewers", "background", "apparition"],
  "pt": ["formulários", "formulário", "hóspede", "hóspedes", "estudioso", "estudiosos", "observador", "observadores", "figura", "figuras", "colono", "colonos", "caracteres", "subcrescimento", "sendo", "marfim", "assinatura", "pescoço", "ver", "vénus"]
}

REPLACE = {
  "en": {},
  "pt": {
    "acidente vascular cerebral": "pincelada",
    "ainda vida": "natureza morta",
  }
}

def get_caption_words(data_path, model="gemma3", lang="en", categories=["all"], return_counts=False):
  with open(data_path, "r") as ifp:
    obj_data = json.load(ifp)

  all_words = {}
  for obj in obj_data.values():
    for obj_cat in categories:
      for raw_word in obj["captions"][model][lang].get(obj_cat, []):
        word = raw_word.lower().replace(".", "")
        word = REPLACE[lang].get(word, word)
        if word not in STOP_WORDS[lang]:
          all_words[word] = all_words.get(word, 0) + 1

  word_cnts = [[w, c] for w,c in all_words.items()]
  words = sorted(word_cnts, key=lambda x:x[1], reverse=True)

  if return_counts:
    return words
  else:
    return [w for w,_ in words]
