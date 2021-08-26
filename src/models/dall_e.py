from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("flax-community/dalle-mini")

model = AutoModelForSeq2SeqLM.from_pretrained("flax-community/dalle-mini")
