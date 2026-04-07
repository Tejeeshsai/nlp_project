from transformers import pipeline

translator = pipeline("translation", model="./my-genz-translator-bart", tokenizer="./my-genz-translator-bart")

slang_text ="homie tryna chill with the crew"
result = translator(slang_text)

print(f"\nSlang: {slang_text}")
print(f"Translation: {result[0]['translation_text']}")
