from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
prompt = "Summarize: The quick brown fox jumps over the lazy dog. It was a sunny day."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("✨ Output:", generated_text)
