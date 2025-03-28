from transformers import pipeline
from warnings import simplefilter

simplefilter(action="ignore")

class EnPt:
  MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-pt"
  pipeline = pipeline(model=MODEL_NAME, device="cuda")

  def translate(txt_en):
    to_pt = ">>por<< " + txt_en
    return EnPt.pipeline(to_pt)[0]["translation_text"]

class PtEn:
  # tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/translation-pt-en-t5")
  # model = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/translation-pt-en-t5")
  # pten_pipeline = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

  # https://huggingface.co/Narrativa/mbart-large-50-finetuned-opus-pt-en-translation

  MODEL_NAME = "unicamp-dl/translation-pt-en-t5"
  pipeline = pipeline(model=MODEL_NAME, device="cuda", max_length=1024)

  def translate(txt_pt):
    to_en = "translate Portuguese to English: " + txt_pt
    return PtEn.pipeline(to_en)[0]["translation_text"]
