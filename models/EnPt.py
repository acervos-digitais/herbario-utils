from transformers import pipeline
from warnings import simplefilter

simplefilter(action="ignore")

class EnPt:
  MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-pt"
  pipeline = pipeline(model=MODEL_NAME, device="cuda")

  def translate(txt_en):
    to_pt = ">>por<< " + txt_en
    return EnPt.pipeline(to_pt)[0]["translation_text"]
