from torch import cuda, no_grad
from transformers import pipeline
from transformers import MarianMTModel, MarianTokenizer
from warnings import simplefilter

simplefilter(action="ignore")

class EnPt:
  MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-pt"
  DEVICE = "cuda" if cuda.is_available() else "cpu"

  def __init__(self):
    self.model = MarianMTModel.from_pretrained(EnPt.MODEL_NAME).to(EnPt.DEVICE)
    self.tokenizer = MarianTokenizer.from_pretrained(EnPt.MODEL_NAME)

  def translate(self, txt_en):
    to_pt = ">>por<< " + txt_en
    inputs = self.tokenizer(to_pt, return_tensors="pt", padding=True).to(self.model.device)

    txt_pt = txt_en
    with no_grad():
      outputs = self.model.generate(**inputs)
      txt_pt = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return txt_pt

class PtEn:
  # tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/translation-pt-en-t5")
  # model = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/translation-pt-en-t5")
  # pten_pipeline = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

  # https://huggingface.co/Narrativa/mbart-large-50-finetuned-opus-pt-en-translation

  MODEL_NAME = "unicamp-dl/translation-pt-en-t5"

  def __init__(self):
    self.pipeline = pipeline(model=PtEn.MODEL_NAME, device="cuda", max_length=1024)

  def translate(self, txt_pt):
    to_en = "translate Portuguese to English: " + txt_pt
    return self.pipeline(to_en)[0]["translation_text"]

class PartOfSpeech:
  MODEL_NAME = "QCRI/bert-base-multilingual-cased-pos-english"

  def __init__(self):
    self.pipeline = pipeline(model=PartOfSpeech.MODEL_NAME, device="cuda")

  def get_nouns(self, txt):
    if txt == "" or len(txt) < 1:
      return []

    pos = self.pipeline(txt)

    nouns = []
    for o in pos:
      if o["entity"] == "NN" or o["entity"] == "NNS":
        if o["word"].startswith("#") and len(nouns) > 1:
          nouns[-1] = nouns[-1] + o["word"].replace("#", "").lower()
        elif not o["word"].startswith("#"):
          nouns.append(o["word"].lower())

    return list(set(nouns))
